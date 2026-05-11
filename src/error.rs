//! Application-wide error type.
//!
//! `AppError` flows through the entire binary. It implements `IntoResponse`
//! so handlers can return `Result<T, AppError>` and get a uniform JSON error
//! body with the right HTTP status.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("migration error: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("hugging face error: {0}")]
    HfHub(String),

    #[error("invalid uuid: {0}")]
    Uuid(#[from] uuid::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("session not found")]
    SessionNotFound,

    #[error("memory not found")]
    MemoryNotFound,

    #[error("model not loaded")]
    ModelNotLoaded,

    #[error("invalid request: {0}")]
    BadRequest(String),

    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("inference error: {0}")]
    Inference(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    fn code(&self) -> &'static str {
        match self {
            AppError::Config(_) => "config_error",
            AppError::Io(_) => "io_error",
            AppError::Database(_) | AppError::Migrate(_) => "database_error",
            AppError::Http(_) | AppError::HfHub(_) => "network_error",
            AppError::Uuid(_) => "invalid_uuid",
            AppError::Json(_) => "json_error",
            AppError::SessionNotFound => "session_not_found",
            AppError::MemoryNotFound => "memory_not_found",
            AppError::ModelNotLoaded => "model_not_loaded",
            AppError::BadRequest(_) => "bad_request",
            AppError::ChecksumMismatch { .. } => "checksum_mismatch",
            AppError::Inference(_) => "inference_error",
            AppError::Internal(_) => "internal_error",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            AppError::SessionNotFound | AppError::MemoryNotFound => StatusCode::NOT_FOUND,
            AppError::BadRequest(_) | AppError::Uuid(_) => StatusCode::BAD_REQUEST,
            AppError::ModelNotLoaded => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = Json(json!({
            "error": {
                "code": self.code(),
                "message": self.to_string(),
            }
        }));
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        } else {
            tracing::debug!(error = %self, "request rejected");
        }
        (status, body).into_response()
    }
}
