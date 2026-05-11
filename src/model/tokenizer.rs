//! Tokenizer helpers.
//!
//! Thin wrapper around the engine's tokenizer for use by the context manager.
//! Kept as a separate module so callers can ignore the engine's wider surface.

use crate::error::AppError;
use crate::model::engine::SharedEngine;

pub async fn count(engine: &SharedEngine, text: &str) -> Result<usize, AppError> {
    engine.count_tokens(text).await
}
