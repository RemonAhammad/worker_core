//! Shared application state passed to every handler via `axum::extract::State`.

use std::sync::Arc;

use sqlx::SqlitePool;

use crate::config::Settings;
use crate::model::engine::SharedEngine;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub db: SqlitePool,
    pub engine: SharedEngine,
}
