//! Agentic chat endpoints.
//!
//! Two endpoints implement a single multi-turn tool-call loop on top of an
//! existing session:
//!
//! - `POST /v1/sessions/:id/agent`          — start a new user turn.
//! - `POST /v1/sessions/:id/agent/continue` — return tool results so the
//!   model can keep going.
//!
//! Each call returns one of:
//!
//! - `AgentResponse::Message`    — the model produced a final reply.
//! - `AgentResponse::ToolCalls`  — the model wants tools; client runs them
//!                                 and posts results back here.
//!
//! Tool calls and tool results are persisted as messages (roles `assistant`
//! and `tool`) so the existing history-trim policy keeps working unchanged.

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
use crate::model::engine::{ChatTurn, GenerateOpts, SharedEngine};
use crate::state::AppState;
use crate::tools::{
    self, ParsedToolCall, filesystem_tools, render_tool_preamble, render_tool_response,
};
use crate::types::{AgentContinueRequest, AgentResponse, AgentSendRequest, Role, Usage};

/// Stop sequence used during agent inference. Hitting the closing
/// `</tool_call>` tag means we can dispatch immediately instead of
/// draining to EOS.
const TOOL_CALL_STOP: &str = "</tool_call>";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/sessions/{id}/agent", post(send))
        .route("/v1/sessions/{id}/agent/continue", post(continue_))
}

async fn send(
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
    Json(req): Json<AgentSendRequest>,
) -> Result<Json<AgentResponse>, AppError> {
    tracing::info!(
        %session_id,
        workspace_hint = ?req.workspace_hint,
        content_len = req.content.len(),
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        "AGENT /agent: send"
    );
    if req.content.trim().is_empty() {
        return Err(AppError::BadRequest("content is required".into()));
    }
    if req.max_tokens == 0 {
        return Err(AppError::BadRequest("max_tokens must be > 0".into()));
    }

    let session = sess_db::get(&state.db, session_id).await?;
    // Respect any pinned model on this session.
    crate::api::models::ensure_model_loaded(&state, &session.model_name).await?;
    let engine = state.engine.current().await;

    // Persist the new user message before generation so the prompt-builder
    // sees it as the trailing turn.
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

    run_turn(
        state,
        engine,
        session_id,
        &session.system_prompt,
        req.max_tokens,
        req.temperature,
        req.workspace_hint.as_deref(),
    )
    .await
}

async fn continue_(
    State(state): State<AppState>,
    Path(session_id): Path<Uuid>,
    Json(req): Json<AgentContinueRequest>,
) -> Result<Json<AgentResponse>, AppError> {
    tracing::info!(
        %session_id,
        n_results = req.results.len(),
        "AGENT /agent/continue: send"
    );
    if req.max_tokens == 0 {
        return Err(AppError::BadRequest("max_tokens must be > 0".into()));
    }

    let session = sess_db::get(&state.db, session_id).await?;
    crate::api::models::ensure_model_loaded(&state, &session.model_name).await?;
    let engine = state.engine.current().await;

    // Persist each tool result as a tool message so the next prompt
    // includes them as conversation history.
    for result in &req.results {
        let payload = render_tool_response(result);
        let tokens = engine.count_tokens(&payload).await? as i64;
        msg_db::insert(&state.db, session_id, Role::Tool, &payload, tokens).await?;
    }

    run_turn(
        state,
        engine,
        session_id,
        &session.system_prompt,
        req.max_tokens,
        req.temperature,
        req.workspace_hint.as_deref(),
    )
    .await
}

/// Run one generation pass against the model, parse the output, and decide
/// whether to return a `Message` or `ToolCalls`.
async fn run_turn(
    state: AppState,
    engine: SharedEngine,
    session_id: Uuid,
    base_system_prompt: &Option<String>,
    max_tokens: u32,
    temperature: f32,
    workspace_hint: Option<&str>,
) -> Result<Json<AgentResponse>, AppError> {
    // Reload the session so we get the freshly-touched updated_at and any
    // history additions from the caller.
    let session = sess_db::get(&state.db, session_id).await?;
    let _ = base_system_prompt; // session.system_prompt is the source of truth

    // The tool catalog goes at the very TOP of the system block (above
    // memories and the session's own prompt) so the model can't miss it.
    let tools = filesystem_tools();
    let preamble = render_tool_preamble(&tools, workspace_hint);

    let cm = ContextManager::new(&engine, &state.db);
    let turns: Vec<ChatTurn> = cm
        .build_with(
            &session,
            engine.context_length(),
            Some(max_tokens),
            Some(&preamble),
        )
        .await?;

    // Dump the prompt fed to the model so we can see exactly what arrives.
    tracing::info!(
        n_turns = turns.len(),
        roles = ?turns.iter().map(|t| t.role.as_str()).collect::<Vec<_>>(),
        "AGENT: prompt assembled"
    );
    if let Some(sys) = turns.iter().find(|t| matches!(t.role, crate::types::Role::System)) {
        let head = sys.content.chars().take(800).collect::<String>();
        tracing::info!(system_chars = sys.content.len(), system_head = %head, "AGENT: system message");
    } else {
        tracing::warn!("AGENT: NO system message in prompt — model has no tool catalog");
    }

    let started = Instant::now();
    let generated = engine
        .generate_with(
            &turns,
            max_tokens,
            temperature,
            GenerateOpts {
                stop_sequences: vec![TOOL_CALL_STOP.to_string()],
            },
        )
        .await?;
    let elapsed = started.elapsed();
    tracing::info!(
        session_id = %session_id,
        prompt_tokens = generated.prompt_tokens,
        completion_tokens = generated.completion_tokens,
        elapsed_ms = elapsed.as_millis() as u64,
        agent = true,
        "AGENT: inference complete"
    );
    // Raw model output, capped for log readability.
    let head = generated.text.chars().take(1200).collect::<String>();
    tracing::info!(raw_chars = generated.text.len(), raw_head = %head, "AGENT: raw model output");

    let outcome = tools::parser::parse(&generated.text);
    tracing::info!(
        n_tool_calls = outcome.tool_calls.len(),
        tool_names = ?outcome.tool_calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        prose_len = outcome.prose.len(),
        "AGENT: parsed outcome"
    );

    let usage = Usage {
        prompt_tokens: generated.prompt_tokens,
        completion_tokens: generated.completion_tokens,
        total_tokens: generated.prompt_tokens + generated.completion_tokens,
    };

    if outcome.tool_calls.is_empty() {
        // Plain assistant message. Persist and return.
        let assistant = msg_db::insert(
            &state.db,
            session_id,
            Role::Assistant,
            &outcome.prose,
            generated.completion_tokens as i64,
        )
        .await?;
        sess_db::touch(&state.db, session_id).await?;
        return Ok(Json(AgentResponse::Message {
            message: assistant,
            usage,
        }));
    }

    // Tool-call branch: persist the assistant message containing the prose
    // AND the literal `<tool_call>` markup, so future prompts replay it
    // faithfully and the model's continuation sees its own prior turn.
    let assistant_content = build_assistant_with_calls(&outcome.prose, &outcome.tool_calls);
    let assistant = msg_db::insert(
        &state.db,
        session_id,
        Role::Assistant,
        &assistant_content,
        generated.completion_tokens as i64,
    )
    .await?;
    sess_db::touch(&state.db, session_id).await?;

    Ok(Json(AgentResponse::ToolCalls {
        assistant_id: assistant.id,
        calls: outcome.tool_calls,
        prose: outcome.prose,
        usage,
    }))
}

/// Re-emit the assistant turn including the `<tool_call>` blocks so the
/// model's next prompt is internally consistent.
fn build_assistant_with_calls(prose: &str, calls: &[ParsedToolCall]) -> String {
    let mut s = String::new();
    if !prose.is_empty() {
        s.push_str(prose);
        s.push('\n');
    }
    for call in calls {
        s.push_str("<tool_call>\n");
        // Re-serialize so the JSON is canonical regardless of what the
        // model emitted (handles weird whitespace / quoting).
        let body = serde_json::json!({
            "name": call.name,
            "arguments": call.arguments,
        });
        s.push_str(
            &serde_json::to_string(&body).unwrap_or_else(|_| "{}".to_string()),
        );
        s.push_str("\n</tool_call>\n");
    }
    s.trim().to_string()
}
