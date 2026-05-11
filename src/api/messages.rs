//! `POST /v1/sessions/:id/messages` — append a user message and synchronously
//! generate the assistant's reply.

use std::time::Instant;

use axum::{
    Json, Router,
    extract::{Path, State},
    routing::post,
};
use uuid::Uuid;

use crate::context::ContextManager;
use crate::db::{messages as msg_db, sessions as sess_db};
use crate::error::AppError;
use crate::memory;
use crate::state::AppState;
use crate::types::{CreateMessageRequest, MessageResponse, Role, Usage};

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/sessions/{id}/messages", post(create))
}

async fn create(
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
    Json(req): Json<CreateMessageRequest>,
) -> Result<Json<MessageResponse>, AppError> {
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content is required".into()));
    }
    if req.max_tokens == 0 {
        return Err(AppError::BadRequest("max_tokens must be > 0".into()));
    }

    let session = sess_db::get(&state.db, session_id).await?;

    let user_token_count = state.engine.count_tokens(&req.content).await? as i64;
    msg_db::insert(
        &state.db,
        session_id,
        Role::User,
        &req.content,
        user_token_count,
    )
    .await?;

    // Best-effort: scan the user turn for self-introduction facts ("my name
    // is X", "i live in Y") and persist them so they survive trims + new
    // sessions. Failures here never block generation.
    if let Err(e) = memory::extract_and_store(&state.db, &req.content).await {
        tracing::warn!(error = %e, "auto-memory extraction failed");
    }

    let cm = ContextManager::new(&state.engine, &state.db);
    let turns = cm
        .build(&session, state.engine.context_length(), Some(req.max_tokens))
        .await?;

    let started = Instant::now();
    let generated = state
        .engine
        .generate(&turns, req.max_tokens, req.temperature)
        .await?;
    let elapsed = started.elapsed();
    tracing::info!(
        session_id = %session_id,
        prompt_tokens = generated.prompt_tokens,
        completion_tokens = generated.completion_tokens,
        elapsed_ms = elapsed.as_millis() as u64,
        "inference complete"
    );

    let assistant = msg_db::insert(
        &state.db,
        session_id,
        Role::Assistant,
        &generated.text,
        generated.completion_tokens as i64,
    )
    .await?;
    sess_db::touch(&state.db, session_id).await?;

    Ok(Json(MessageResponse {
        message: assistant,
        usage: Usage {
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            total_tokens: generated.prompt_tokens + generated.completion_tokens,
        },
    }))
}
