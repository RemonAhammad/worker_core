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
