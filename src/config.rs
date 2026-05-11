//! Application configuration.
//!
//! Configuration is loaded from `config.toml` if present, then layered with
//! environment variables prefixed `LLM_BACKEND_` (nested keys use `__` as the
//! separator). Reasonable defaults are provided so the binary can run without
//! any config file at all.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::AppError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "Settings::default_models_dir")]
    pub models_dir: PathBuf,
    /// If set, the named preset from `crate::model::presets` replaces the
    /// `[model]` section at startup. CLI `--preset` overrides this.
    #[serde(default)]
    pub model_preset: Option<String>,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
}

impl Settings {
    fn default_models_dir() -> PathBuf {
        PathBuf::from("./models")
    }

    /// Load configuration from `path` (if present) and overlay env vars.
    pub fn load(path: &str) -> Result<Self, AppError> {
        let mut builder = config::Config::builder()
            .set_default("models_dir", "./models")?
            .set_default("server.host", "0.0.0.0")?
            .set_default("server.port", 6969_i64)?
            .set_default("model.repo", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")?
            .set_default("model.filename", "qwen2.5-coder-7b-instruct-q4_k_m.gguf")?
            .set_default("model.sha256", "")?
            .set_default("model.context_length", 8192_i64)?
            .set_default("model.gpu_layers", -1_i64)?
            .set_default("database.path", "./data/backend.db")?;

        if std::path::Path::new(path).exists() {
            builder = builder.add_source(config::File::with_name(path));
        }

        builder = builder.add_source(
            config::Environment::with_prefix("LLM_BACKEND")
                .separator("__")
                .try_parsing(true),
        );

        let cfg = builder.build()?;
        let settings: Settings = cfg.try_deserialize()?;
        Ok(settings)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 6969,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub repo: String,
    pub filename: String,
    #[serde(default)]
    pub sha256: String,
    pub context_length: u32,
    /// -1 = all layers on GPU, 0 = CPU only, N = first N layers on GPU.
    pub gpu_layers: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".into(),
            filename: "qwen2.5-coder-7b-instruct-q4_k_m.gguf".into(),
            sha256: String::new(),
            context_length: 8192,
            gpu_layers: -1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub path: PathBuf,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./data/backend.db"),
        }
    }
}
