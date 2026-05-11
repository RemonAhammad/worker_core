//! Shared request and response types for the HTTP API.
//!
//! Kept deliberately close to OpenAI's chat-completions shape so a generic
//! client can target either this backend or OpenAI without large adapters.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[serde(rename_all = "lowercase")]
#[sqlx(rename_all = "lowercase")]
#[sqlx(type_name = "TEXT")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub title: String,
    pub model_name: String,
    pub system_prompt: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: Role,
    pub content: String,
    pub token_count: i64,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionWithMessages {
    #[serde(flatten)]
    pub session: Session,
    pub messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub title: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ListSessionsQuery {
    #[serde(default = "default_limit")]
    pub limit: i64,
    #[serde(default)]
    pub offset: i64,
}

fn default_limit() -> i64 {
    20
}

#[derive(Debug, Deserialize)]
pub struct CreateMessageRequest {
    pub content: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    1024
}

fn default_temperature() -> f32 {
    0.7
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: Message,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model: String,
    pub loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub size_bytes: u64,
    pub loaded: bool,
}

#[derive(Debug, Serialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

/// Rich model catalog returned by `GET /v1/models/catalog`. Unifies the
/// curated preset list with whatever is on disk so the UI can render a
/// single switchable list.
#[derive(Debug, Serialize)]
pub struct ModelCatalog {
    /// Filename of the currently loaded GGUF (matches `entries[i].filename`
    /// for the loaded entry).
    pub current: String,
    pub entries: Vec<ModelCatalogEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelCatalogEntry {
    /// Stable identifier the caller passes back to `/v1/models/load`.
    /// For presets this is the preset name (e.g. `qwen-coder-14b`); for
    /// local-only files it is the filename.
    pub name: String,
    #[serde(rename = "kind")]
    pub kind: ModelKind,
    pub repo: String,
    pub filename: String,
    pub context_length: u32,
    /// Size of the GGUF on disk if present, `None` if not yet downloaded.
    pub size_bytes: Option<u64>,
    pub min_ram_gib: Option<u32>,
    pub description: Option<String>,
    pub present: bool,
    pub loaded: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Preset,
    Local,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub name: String,
}

/// Body for `PATCH /v1/sessions/:id`. Every field is optional; only the
/// supplied ones are updated.
#[derive(Debug, Deserialize)]
pub struct UpdateSessionRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Preferred GGUF filename for this session. When set, message/agent
    /// handlers will hot-swap the engine to this model on the next send
    /// if the currently-loaded model differs.
    #[serde(default)]
    pub model_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Agent endpoint — multi-turn tool-call loop.
// ---------------------------------------------------------------------------

/// Body of `POST /v1/sessions/:id/agent` — starts a new agent turn.
#[derive(Debug, Deserialize)]
pub struct AgentSendRequest {
    pub content: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Workspace root the client is operating in. Informational — the
    /// backend includes it in the system prompt so the model knows what
    /// it can touch. Tools execute on the client, so the backend has no
    /// way to enforce this itself.
    #[serde(default)]
    pub workspace_hint: Option<String>,
}

/// Body of `POST /v1/sessions/:id/agent/continue` — caller returns tool
/// results from the previous turn so the model can continue.
#[derive(Debug, Deserialize)]
pub struct AgentContinueRequest {
    /// Echoes the `assistant_id` from the previous tool-calls response;
    /// kept for clarity and future correlation, currently unused.
    #[serde(default)]
    pub assistant_id: Option<Uuid>,
    pub results: Vec<crate::tools::ToolResult>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub workspace_hint: Option<String>,
}

/// One of two terminal outcomes for an agent turn.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentResponse {
    /// Model produced a final user-facing message; the loop is over for
    /// this turn.
    Message {
        message: Message,
        usage: Usage,
    },
    /// Model wants to call tool(s). Caller executes them and POSTs the
    /// results back via `/agent/continue` referencing `assistant_id`.
    ToolCalls {
        /// Id of the (partial) assistant message persisted alongside the
        /// tool call so the next turn can append onto it.
        assistant_id: Uuid,
        calls: Vec<crate::tools::ParsedToolCall>,
        /// Prose the model emitted before the tool call(s); already saved
        /// as the assistant message content.
        prose: String,
        usage: Usage,
    },
}

/// A long-term fact that should survive across all sessions. Injected into
/// every prompt's system block at context-build time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: Uuid,
    pub content: String,
    /// "manual" for user-added, "auto" for regex-extracted.
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct CreateMemoryRequest {
    pub content: String,
}

/// Request body for the sticky-session endpoint. No session id required —
/// the server reuses the most-recently-touched session (or creates one).
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub content: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Used only when the server has to create a fresh session.
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub session_id: Uuid,
    pub message: Message,
    pub usage: Usage,
}

/// Returned by `GET /v1/sessions/:id/debug`. Mirrors exactly what would be
/// sent to the model for the *next* generation call on this session.
#[derive(Debug, Serialize)]
pub struct DebugTurn {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct DebugContext {
    pub session_id: Uuid,
    pub context_length: u32,
    pub turns: Vec<DebugTurn>,
    pub prompt_tokens_estimate: u32,
    pub memories_injected: usize,
}
