//! Session CRUD: `/v1/sessions` and `/v1/sessions/:id`.

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use uuid::Uuid;

use crate::context::ContextManager;
use crate::db::{messages as msg_db, sessions as sess_db};
use crate::error::AppError;
use crate::state::AppState;
use crate::types::{
    CreateSessionRequest, DebugContext, DebugTurn, ListSessionsQuery, Session,
    SessionWithMessages,
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/sessions", post(create).get(list))
        .route("/v1/sessions/{id}", get(get_one).delete(delete))
        .route("/v1/sessions/{id}/debug", get(debug))
}

async fn create(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<(StatusCode, Json<Session>), AppError> {
    if req.title.trim().is_empty() {
        return Err(AppError::BadRequest("title is required".into()));
    }
    let session = sess_db::create(
        &state.db,
        req.title.trim(),
        state.engine.model_name(),
        req.system_prompt.as_deref(),
    )
    .await?;
    Ok((StatusCode::CREATED, Json(session)))
}

async fn list(
    State(state): State<AppState>,
    Query(q): Query<ListSessionsQuery>,
) -> Result<Json<Vec<Session>>, AppError> {
    let limit = q.limit.clamp(1, 200);
    let offset = q.offset.max(0);
    let sessions = sess_db::list(&state.db, limit, offset).await?;
    Ok(Json(sessions))
}

async fn get_one(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionWithMessages>, AppError> {
    let session = sess_db::get(&state.db, id).await?;
    let messages = msg_db::list_for_session(&state.db, id).await?;
    Ok(Json(SessionWithMessages { session, messages }))
}

async fn delete(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, AppError> {
    sess_db::delete(&state.db, id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Returns the exact list of chat turns that would be sent to the model for
/// the next generation call on this session — system prompt, injected
/// memories, and the trimmed history. Use this to diagnose "the model forgot
/// X" complaints without rerunning generation.
async fn debug(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<DebugContext>, AppError> {
    let session = sess_db::get(&state.db, id).await?;
    let cm = ContextManager::new(&state.engine, &state.db);
    let turns = cm
        .build(&session, state.engine.context_length(), None)
        .await?;

    let mut prompt_tokens_estimate: u32 = 0;
    let mut debug_turns = Vec::with_capacity(turns.len());
    let mut memories_injected = 0usize;
    for t in &turns {
        prompt_tokens_estimate += state.engine.count_tokens(&t.content).await? as u32;
        if matches!(t.role, crate::types::Role::System) {
            memories_injected = t.content.matches("\n- ").count();
        }
        debug_turns.push(DebugTurn {
            role: t.role,
            content: t.content.clone(),
        });
    }

    Ok(Json(DebugContext {
        session_id: id,
        context_length: state.engine.context_length(),
        turns: debug_turns,
        prompt_tokens_estimate,
        memories_injected,
    }))
}
