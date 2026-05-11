//! Database layer.
//!
//! Owns the SQLite connection pool, runs migrations on startup, and exposes
//! per-table CRUD modules (`sessions`, `messages`).

use std::path::Path;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};

use crate::error::AppError;

pub mod memories;
pub mod messages;
pub mod sessions;

/// Open the SQLite pool, creating the database file (and parent dirs) if
/// missing, then apply embedded migrations.
pub async fn init(path: &Path) -> Result<SqlitePool, AppError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await?;
    }

    let opts = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Wal)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(8)
        .min_connections(1)
        .connect_with(opts)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;

    // Confirm the pool is healthy before returning. Surfacing any
    // initialization failure here is much friendlier than failing at the
    // first request.
    sqlx::query("SELECT 1").execute(&pool).await?;

    Ok(pool)
}
