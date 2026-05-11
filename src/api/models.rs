//! `GET /v1/models` — list GGUF files in the configured `models_dir` and
//! flag which one is currently loaded.

use axum::{Json, Router, extract::State, routing::get};

use crate::error::AppError;
use crate::model::downloader;
use crate::state::AppState;
use crate::types::{ListModelsResponse, ModelInfo};

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/models", get(list_models))
}

async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<ListModelsResponse>, AppError> {
    let entries = downloader::list_local_models(&state.settings.models_dir).await?;
    let loaded = state.engine.model_name();
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
