//! Memories CRUD: `/v1/memories` and `/v1/memories/:id`.
//!
//! Memories are facts the assistant should remember across every session
//! (e.g. "user's name is Rimon"). They are injected into the system prompt
//! by `ContextManager::build`.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete as delete_route, get},
};
use uuid::Uuid;

use crate::db::memories as mem_db;
use crate::error::AppError;
use crate::state::AppState;
use crate::types::{CreateMemoryRequest, Memory};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/memories", get(list).post(create))
        .route("/v1/memories/{id}", delete_route(delete))
}

async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateMemoryRequest>,
) -> Result<(StatusCode, Json<Memory>), AppError> {
    let content = req.content.trim();
    if content.is_empty() {
        return Err(AppError::BadRequest("content is required".into()));
    }
    let memory = mem_db::upsert(&state.db, content, "manual").await?;
    Ok((StatusCode::CREATED, Json(memory)))
}

async fn list(State(state): State<AppState>) -> Result<Json<Vec<Memory>>, AppError> {
    let memories = mem_db::list(&state.db).await?;
    Ok(Json(memories))
}

async fn delete(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    mem_db::delete(&state.db, id).await?;
    Ok(StatusCode::NO_CONTENT)
}
