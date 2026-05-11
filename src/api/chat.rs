//! Sticky-session chat endpoint: `POST /v1/chat`.
//!
//! The caller does NOT pass a session id. The server reuses the most
//! recently-updated session, or creates a fresh one if none exists. This is
//! what most clients actually want — a single ongoing conversation — and
//! removes the most common cause of "the model forgot what I said": each
//! request landing in a brand-new session.

use std::time::Instant;

use axum::{Json, Router, extract::State, routing::post};

use crate::context::ContextManager;
use crate::db::{messages as msg_db, sessions as sess_db};
use crate::error::AppError;
use crate::memory;
use crate::state::AppState;
use crate::types::{ChatRequest, ChatResponse, Role, Usage};

pub fn router() -> Router<AppState> {
    Router::new().route("/v1/chat", post(chat))
}

async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content is required".into()));
    }
    if req.max_tokens == 0 {
        return Err(AppError::BadRequest("max_tokens must be > 0".into()));
    }

    // Pick up the most recent session, or create one. We deliberately do
    // not maintain a separate "current session" pointer — `updated_at` on
    // sessions is already the source of truth, and using it keeps `/v1/chat`
    // stateless and survives restarts.
    let session = match sess_db::most_recent(&state.db).await? {
        Some(s) => s,
        None => {
            let engine = state.engine.current().await;
            sess_db::create(
                &state.db,
                "chat",
                engine.model_name(),
                req.system_prompt.as_deref(),
            )
            .await?
        }
    };

    // Respect any per-session model pin before snapshotting the engine.
    crate::api::models::ensure_model_loaded(&state, &session.model_name).await?;
    let engine = state.engine.current().await;

    let user_token_count = engine.count_tokens(&req.content).await? as i64;
    msg_db::insert(
        &state.db,
        session.id,
        Role::User,
        &req.content,
        user_token_count,
    )
    .await?;

    if let Err(e) = memory::extract_and_store(&state.db, &req.content).await {
        tracing::warn!(error = %e, "auto-memory extraction failed");
    }

    let cm = ContextManager::new(&engine, &state.db);
    let turns = cm
        .build(&session, engine.context_length(), Some(req.max_tokens))
        .await?;

    let started = Instant::now();
    let generated = engine
        .generate(&turns, req.max_tokens, req.temperature)
        .await?;
    let elapsed = started.elapsed();
    tracing::info!(
        session_id = %session.id,
        prompt_tokens = generated.prompt_tokens,
        completion_tokens = generated.completion_tokens,
        elapsed_ms = elapsed.as_millis() as u64,
        sticky = true,
        "inference complete"
    );

    let assistant = msg_db::insert(
        &state.db,
        session.id,
        Role::Assistant,
        &generated.text,
        generated.completion_tokens as i64,
    )
    .await?;
    sess_db::touch(&state.db, session.id).await?;

    Ok(Json(ChatResponse {
        session_id: session.id,
        message: assistant,
        usage: Usage {
            prompt_tokens: generated.prompt_tokens,
            completion_tokens: generated.completion_tokens,
            total_tokens: generated.prompt_tokens + generated.completion_tokens,
        },
    }))
}
