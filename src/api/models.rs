//! `/v1/models` family of endpoints.
//!
//! - `GET /v1/models`           — flat list of GGUFs on disk (legacy shape).
//! - `GET /v1/models/catalog`   — merged catalog: every preset + every local
//!                                file, each tagged with present/loaded.
//! - `POST /v1/models/load`     — hot-swap the active model (downloads first
//!                                if the GGUF isn't on disk yet).

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};

use crate::error::AppError;
use crate::model::engine::{LlamaEngine, SharedEngine};
use crate::model::{downloader, presets};
use crate::state::AppState;
use crate::types::{
    ListModelsResponse, LoadModelRequest, ModelCatalog, ModelCatalogEntry, ModelInfo, ModelKind,
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/catalog", get(catalog))
        .route("/v1/models/load", post(load_model))
}

async fn list_models(State(state): State<AppState>) -> Result<Json<ListModelsResponse>, AppError> {
    let entries = downloader::list_local_models(&state.settings.models_dir).await?;
    let engine = state.engine.current().await;
    let loaded = engine.model_name().to_string();
    let models = entries
        .into_iter()
        .map(|(name, size_bytes)| ModelInfo {
            loaded: name == loaded,
            name,
            size_bytes,
        })
        .collect();
    Ok(Json(ListModelsResponse { models }))
}

async fn catalog(State(state): State<AppState>) -> Result<Json<ModelCatalog>, AppError> {
    let local_files = downloader::list_local_models(&state.settings.models_dir).await?;
    let engine = state.engine.current().await;
    let loaded = engine.model_name().to_string();
    let mut entries: Vec<ModelCatalogEntry> = Vec::new();

    // Curated presets first.
    for p in presets::PRESETS {
        let local = local_files.iter().find(|(name, _)| name == p.filename);
        entries.push(ModelCatalogEntry {
            name: p.name.to_string(),
            kind: ModelKind::Preset,
            repo: p.repo.to_string(),
            filename: p.filename.to_string(),
            context_length: p.context_length,
            size_bytes: local.map(|(_, s)| *s),
            min_ram_gib: Some(p.min_ram_gib),
            description: Some(p.description.to_string()),
            present: local.is_some(),
            loaded: loaded == p.filename,
        });
    }

    // Local GGUFs that aren't in the preset catalog.
    for (name, size) in &local_files {
        if entries.iter().any(|e| e.filename == *name) {
            continue;
        }
        entries.push(ModelCatalogEntry {
            name: name.clone(),
            kind: ModelKind::Local,
            repo: String::new(),
            filename: name.clone(),
            context_length: state.settings.model.context_length,
            size_bytes: Some(*size),
            min_ram_gib: None,
            description: None,
            present: true,
            loaded: loaded == *name,
        });
    }

    Ok(Json(ModelCatalog {
        current: loaded,
        entries,
    }))
}

async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<ModelCatalogEntry>, AppError> {
    let backend = state
        .llama_backend
        .as_ref()
        .ok_or_else(|| {
            AppError::Internal(
                "model loading is not available when running with the stub backend".into(),
            )
        })?
        .clone();

    // Resolve name → (repo, filename, context_length, kind, preset_ref).
    let preset = presets::find(&req.name);
    let (repo, filename, context_length, kind, min_ram, description) = match preset {
        Some(p) => (
            p.repo.to_string(),
            p.filename.to_string(),
            p.context_length,
            ModelKind::Preset,
            Some(p.min_ram_gib),
            Some(p.description.to_string()),
        ),
        None => {
            // Treat the name as a filename of a local GGUF.
            let local_path = state.settings.models_dir.join(&req.name);
            if !local_path.exists() {
                return Err(AppError::BadRequest(format!(
                    "unknown model: {} (no matching preset and no local file)",
                    req.name
                )));
            }
            (
                String::new(),
                req.name.clone(),
                state.settings.model.context_length,
                ModelKind::Local,
                None,
                None,
            )
        }
    };

    // If this exact GGUF is already loaded, short-circuit.
    {
        let current = state.engine.current().await;
        if current.model_name() == filename {
            let size_bytes = tokio::fs::metadata(state.settings.models_dir.join(&filename))
                .await
                .ok()
                .map(|m| m.len());
            return Ok(Json(ModelCatalogEntry {
                name: req.name.clone(),
                kind,
                repo,
                filename,
                context_length,
                size_bytes,
                min_ram_gib: min_ram,
                description,
                present: true,
                loaded: true,
            }));
        }
    }

    // Ensure the file is on disk (downloads + resumes if not).
    let path = if !repo.is_empty() {
        let resolved = downloader::ensure_model_present(
            &state.settings.models_dir,
            &repo,
            &filename,
            "",
        )
        .await?;
        resolved.path
    } else {
        state.settings.models_dir.join(&filename)
    };

    let gpu_layers = state.settings.model.gpu_layers;
    tracing::info!(name = %req.name, "loading new model");
    let new_engine = tokio::task::spawn_blocking(move || {
        LlamaEngine::load(backend, &path, context_length, gpu_layers)
    })
    .await
    .map_err(|e| AppError::Internal(format!("model-load task join error: {e}")))??;
    let new_engine: SharedEngine = Arc::new(new_engine);

    // Atomically swap. The old engine drops as soon as the last in-flight
    // request finishes with it.
    let _old = state.engine.replace(new_engine).await;
    tracing::info!(name = %req.name, "model swapped");

    let size_bytes = tokio::fs::metadata(state.settings.models_dir.join(&filename))
        .await
        .ok()
        .map(|m| m.len());
    Ok(Json(ModelCatalogEntry {
        name: req.name.clone(),
        kind,
        repo,
        filename,
        context_length,
        size_bytes,
        min_ram_gib: min_ram,
        description,
        present: true,
        loaded: true,
    }))
}
