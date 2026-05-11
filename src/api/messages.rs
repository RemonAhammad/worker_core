//! `POST /v1/sessions/:id/messages` — append a user message and synchronously
//! generate the assistant's reply.

use std::convert::Infallible;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
};
use futures_util::stream::Stream;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::context::ContextManager;
use crate::db::{messages as msg_db, sessions as sess_db};
use crate::error::AppError;
use crate::memory;
use crate::model::engine::{GenerateOpts, StreamEvent};
use crate::state::AppState;
use crate::types::{CreateMessageRequest, MessageResponse, Role, Usage};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/sessions/{id}/messages", post(create))
        .route("/v1/sessions/{id}/messages/stream", post(create_stream))
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

    // If the session has pinned a specific model, swap it in before we
    // snapshot. The swap is a no-op if already loaded.
    crate::api::models::ensure_model_loaded(&state, &session.model_name).await?;

    // Snapshot the engine once per request so a mid-request hot-swap doesn't
    // mix tokenization from one model with generation by another.
    let engine = state.engine.current().await;

    let user_token_count = engine.count_tokens(&req.content).await? as i64;
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

/// `POST /v1/sessions/:id/messages/stream` — SSE variant.
///
/// Streams events of three kinds:
///   - `{"type":"token","text":"..."}`        — incremental decoded text
///   - `{"type":"done","message":...,"usage":...}` — final persisted message
///   - `{"type":"error","message":"..."}`     — generation or pipeline failure
async fn create_stream(
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
    Json(req): Json<CreateMessageRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content is required".into()));
    }
    if req.max_tokens == 0 {
        return Err(AppError::BadRequest("max_tokens must be > 0".into()));
    }

    let session = sess_db::get(&state.db, session_id).await?;
    crate::api::models::ensure_model_loaded(&state, &session.model_name).await?;
    let engine = state.engine.current().await;

    let user_tokens = engine.count_tokens(&req.content).await? as i64;
    msg_db::insert(
        &state.db,
        session_id,
        Role::User,
        &req.content,
        user_tokens,
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
    let rx = engine
        .generate_streaming(&turns, req.max_tokens, req.temperature, GenerateOpts::default())
        .await?;

    // Splice in a finalization step after the stream ends that persists the
    // assistant message and emits the `done` event with usage + the saved
    // row. We do this by collecting tokens into a local buffer and emitting
    // events as they arrive; on Done we run the DB writes inside a small
    // async block whose result becomes the final SSE event.
    let pool = state.db.clone();
    let stream = async_stream::stream! {
        let mut buf = String::new();
        let mut rx = ReceiverStream::new(rx);
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut errored = false;
        while let Some(event) = rx.next().await {
            match event {
                StreamEvent::Token(t) => {
                    buf.push_str(&t);
                    let data = json!({ "type": "token", "text": t });
                    yield Ok::<_, Infallible>(Event::default().data(data.to_string()));
                }
                StreamEvent::Done { prompt_tokens: p, completion_tokens: c, .. } => {
                    prompt_tokens = p;
                    completion_tokens = c;
                }
                StreamEvent::Error(msg) => {
                    errored = true;
                    let data = json!({ "type": "error", "message": msg });
                    yield Ok(Event::default().data(data.to_string()));
                    break;
                }
            }
        }

        if errored {
            return;
        }

        let elapsed = started.elapsed();
        tracing::info!(
            session_id = %session_id,
            prompt_tokens, completion_tokens,
            elapsed_ms = elapsed.as_millis() as u64,
            streaming = true,
            "inference complete"
        );

        match msg_db::insert(
            &pool,
            session_id,
            Role::Assistant,
            &buf,
            completion_tokens as i64,
        )
        .await
        {
            Ok(assistant) => {
                let _ = sess_db::touch(&pool, session_id).await;
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                };
                let data = json!({
                    "type": "done",
                    "message": assistant,
                    "usage": usage,
                });
                yield Ok(Event::default().data(data.to_string()));
            }
            Err(e) => {
                let data = json!({ "type": "error", "message": format!("persist failed: {e}") });
                yield Ok(Event::default().data(data.to_string()));
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
