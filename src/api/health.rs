//! `GET /health` — liveness + currently-loaded model info.

use axum::{Json, Router, extract::State, routing::get};

use crate::error::AppError;
use crate::state::AppState;
use crate::types::HealthResponse;

pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health))
}

async fn health(State(state): State<AppState>) -> Result<Json<HealthResponse>, AppError> {
    Ok(Json(HealthResponse {
        status: "ok",
        model: state.engine.model_name().to_string(),
        loaded: true,
    }))
}
