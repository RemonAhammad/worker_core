//! co_worker_lite — local LLM backend service.
//!
//! Init order: CLI → config → logging → database → model (download + load) →
//! HTTP server. SIGINT/SIGTERM trigger graceful shutdown which lets in-flight
//! requests finish before the DB pool closes.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use llama_cpp_2::llama_backend::LlamaBackend;
use tokio::net::TcpListener;
use tracing_subscriber::{EnvFilter, fmt};

use co_worker_lite::api;
use co_worker_lite::config::Settings;
use co_worker_lite::db;
use co_worker_lite::model::{
    downloader,
    engine::{LlamaEngine, SharedEngine},
    presets,
};
use co_worker_lite::state::{AppState, EngineSlot};

#[derive(Parser, Debug)]
#[command(
    name = "co_worker_lite",
    version,
    about = "Local LLM backend serving a chat API over HTTP",
    disable_help_subcommand = true
)]
struct Cli {
    /// Pick a curated model preset (overrides `[model]` in config.toml).
    /// Use `--list-presets` to see options.
    #[arg(long, value_name = "NAME")]
    preset: Option<String>,

    /// Print the model preset catalog and exit.
    #[arg(long)]
    list_presets: bool,

    /// Path to the config file (default: ./config.toml; ignored if missing).
    #[arg(long, value_name = "PATH", default_value = "config.toml")]
    config: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.list_presets {
        print!("{}", presets::render_listing());
        return Ok(());
    }

    let mut settings = Settings::load(&cli.config).context("loading configuration")?;

    // Preset selection precedence: CLI flag > config-level `model_preset`.
    let preset_name: Option<String> = cli
        .preset
        .clone()
        .or_else(|| settings.model_preset.clone());
    if let Some(name) = preset_name.as_deref() {
        let preset = presets::find(name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown preset '{name}'. Run `co_worker_lite --list-presets` to see the catalog."
            )
        })?;
        presets::apply(&mut settings, preset);
    }

    init_tracing();

    if let Some(name) = preset_name.as_deref() {
        tracing::info!(preset = name, "model preset applied");
    }
    tracing::info!(
        host = %settings.server.host,
        port = settings.server.port,
        models_dir = %settings.models_dir.display(),
        model_repo = %settings.model.repo,
        model_file = %settings.model.filename,
        "co_worker_lite starting"
    );

    let pool = db::init(&settings.database.path)
        .await
        .context("initializing database")?;

    let resolved = downloader::ensure_model_present(
        &settings.models_dir,
        &settings.model.repo,
        &settings.model.filename,
        &settings.model.sha256,
    )
    .await
    .context("ensuring model is present")?;
    tracing::info!(
        path = %resolved.path.display(),
        size_bytes = resolved.size_bytes,
        "model file ready"
    );

    let backend = Arc::new(
        LlamaBackend::init().map_err(|e| anyhow::anyhow!("llama backend init failed: {e}"))?,
    );
    let engine = LlamaEngine::load(
        backend.clone(),
        &resolved.path,
        settings.model.context_length,
        settings.model.gpu_layers,
    )
    .context("loading model into memory")?;
    let engine: SharedEngine = Arc::new(engine);

    let state = AppState {
        settings: Arc::new(settings.clone()),
        db: pool.clone(),
        engine: EngineSlot::new(engine),
        llama_backend: Some(backend),
    };

    let app = api::router(state);

    let addr: SocketAddr = format!("{}:{}", settings.server.host, settings.server.port)
        .parse()
        .context("invalid server host/port")?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!(%addr, "listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum server error")?;

    tracing::info!("server stopped, closing database");
    pool.close().await;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,co_worker_lite=debug"));
    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_level(true)
        .compact()
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("ctrl-c received, shutting down"),
        _ = terminate => tracing::info!("SIGTERM received, shutting down"),
    }
}
