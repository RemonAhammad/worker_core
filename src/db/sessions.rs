//! CRUD operations for the `sessions` table.

use chrono::Utc;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::error::AppError;
use crate::types::Session;

fn row_to_session(row: &sqlx::sqlite::SqliteRow) -> Result<Session, AppError> {
    let id: String = row.try_get("id")?;
    let metadata: String = row.try_get("metadata")?;
    Ok(Session {
        id: Uuid::parse_str(&id)?,
        title: row.try_get("title")?,
        model_name: row.try_get("model_name")?,
        system_prompt: row.try_get("system_prompt")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        metadata: serde_json::from_str(&metadata)?,
    })
}

pub async fn create(
    pool: &SqlitePool,
    title: &str,
    model_name: &str,
    system_prompt: Option<&str>,
) -> Result<Session, AppError> {
    let id = Uuid::new_v4();
    let now = Utc::now();
    let metadata = "{}".to_string();

    sqlx::query(
        r#"
        INSERT INTO sessions (id, title, model_name, system_prompt, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(id.to_string())
    .bind(title)
    .bind(model_name)
    .bind(system_prompt)
    .bind(now)
    .bind(now)
    .bind(&metadata)
    .execute(pool)
    .await?;

    Ok(Session {
        id,
        title: title.to_string(),
        model_name: model_name.to_string(),
        system_prompt: system_prompt.map(str::to_string),
        created_at: now,
        updated_at: now,
        metadata: serde_json::Value::Object(Default::default()),
    })
}

pub async fn get(pool: &SqlitePool, id: Uuid) -> Result<Session, AppError> {
    let row = sqlx::query(
        r#"SELECT id, title, model_name, system_prompt, created_at, updated_at, metadata
           FROM sessions WHERE id = ?"#,
    )
    .bind(id.to_string())
    .fetch_optional(pool)
    .await?;

    let row = row.ok_or(AppError::SessionNotFound)?;
    row_to_session(&row)
}

pub async fn list(
    pool: &SqlitePool,
    limit: i64,
    offset: i64,
) -> Result<Vec<Session>, AppError> {
    let rows = sqlx::query(
        r#"SELECT id, title, model_name, system_prompt, created_at, updated_at, metadata
           FROM sessions
           ORDER BY created_at DESC
           LIMIT ? OFFSET ?"#,
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await?;

    rows.iter().map(row_to_session).collect()
}

pub async fn delete(pool: &SqlitePool, id: Uuid) -> Result<(), AppError> {
    let result = sqlx::query("DELETE FROM sessions WHERE id = ?")
        .bind(id.to_string())
        .execute(pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::SessionNotFound);
    }
    Ok(())
}

/// Return the session with the most recent `updated_at`, or `None` if no
/// sessions exist. Used by the sticky-session `/v1/chat` endpoint.
pub async fn most_recent(pool: &SqlitePool) -> Result<Option<Session>, AppError> {
    let row = sqlx::query(
        r#"SELECT id, title, model_name, system_prompt, created_at, updated_at, metadata
           FROM sessions
           ORDER BY updated_at DESC
           LIMIT 1"#,
    )
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(row_to_session(&r)?)),
        None => Ok(None),
    }
}

pub async fn touch(pool: &SqlitePool, id: Uuid) -> Result<(), AppError> {
    sqlx::query("UPDATE sessions SET updated_at = ? WHERE id = ?")
        .bind(Utc::now())
        .bind(id.to_string())
        .execute(pool)
        .await?;
    Ok(())
}
