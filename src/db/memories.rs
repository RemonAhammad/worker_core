//! CRUD operations for the `memories` table.
//!
//! Memories are long-term facts that the assistant should remember across
//! every session. They are injected into the system prompt at context-build
//! time by `ContextManager`.

use chrono::Utc;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::error::AppError;
use crate::types::Memory;

fn row_to_memory(row: &sqlx::sqlite::SqliteRow) -> Result<Memory, AppError> {
    let id: String = row.try_get("id")?;
    Ok(Memory {
        id: Uuid::parse_str(&id)?,
        content: row.try_get("content")?,
        source: row.try_get("source")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

/// Insert a new memory. If a row with the same `content` already exists, the
/// existing row is returned untouched (the unique index makes the insert a
/// no-op via `INSERT OR IGNORE`).
pub async fn upsert(
    pool: &SqlitePool,
    content: &str,
    source: &str,
) -> Result<Memory, AppError> {
    let id = Uuid::new_v4();
    let now = Utc::now();

    sqlx::query(
        r#"
        INSERT OR IGNORE INTO memories (id, content, source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        "#,
    )
    .bind(id.to_string())
    .bind(content)
    .bind(source)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await?;

    // Fetch the actual row — could be the one we just inserted, or a
    // pre-existing one with the same content.
    let row = sqlx::query(
        r#"SELECT id, content, source, created_at, updated_at
           FROM memories WHERE content = ?"#,
    )
    .bind(content)
    .fetch_one(pool)
    .await?;

    row_to_memory(&row)
}

pub async fn list(pool: &SqlitePool) -> Result<Vec<Memory>, AppError> {
    let rows = sqlx::query(
        r#"SELECT id, content, source, created_at, updated_at
           FROM memories
           ORDER BY created_at ASC"#,
    )
    .fetch_all(pool)
    .await?;

    rows.iter().map(row_to_memory).collect()
}

pub async fn delete(pool: &SqlitePool, id: Uuid) -> Result<(), AppError> {
    let result = sqlx::query("DELETE FROM memories WHERE id = ?")
        .bind(id.to_string())
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::MemoryNotFound);
    }
    Ok(())
}
