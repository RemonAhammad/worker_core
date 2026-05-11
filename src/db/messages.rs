//! CRUD operations for the `messages` table.

use chrono::Utc;
use sqlx::{Row, SqlitePool};
use uuid::Uuid;

use crate::error::AppError;
use crate::types::{Message, Role};

fn row_to_message(row: &sqlx::sqlite::SqliteRow) -> Result<Message, AppError> {
    let id: String = row.try_get("id")?;
    let session_id: String = row.try_get("session_id")?;
    let role_str: String = row.try_get("role")?;
    let metadata: String = row.try_get("metadata")?;
    let role = match role_str.as_str() {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        other => return Err(AppError::Internal(format!("unknown role: {other}"))),
    };
    Ok(Message {
        id: Uuid::parse_str(&id)?,
        session_id: Uuid::parse_str(&session_id)?,
        role,
        content: row.try_get("content")?,
        token_count: row.try_get("token_count")?,
        created_at: row.try_get("created_at")?,
        metadata: serde_json::from_str(&metadata)?,
    })
}

pub async fn insert(
    pool: &SqlitePool,
    session_id: Uuid,
    role: Role,
    content: &str,
    token_count: i64,
) -> Result<Message, AppError> {
    let id = Uuid::new_v4();
    let now = Utc::now();
    let metadata = "{}".to_string();

    sqlx::query(
        r#"
        INSERT INTO messages (id, session_id, role, content, token_count, created_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(id.to_string())
    .bind(session_id.to_string())
    .bind(role.as_str())
    .bind(content)
    .bind(token_count)
    .bind(now)
    .bind(&metadata)
    .execute(pool)
    .await?;

    Ok(Message {
        id,
        session_id,
        role,
        content: content.to_string(),
        token_count,
        created_at: now,
        metadata: serde_json::Value::Object(Default::default()),
    })
}

pub async fn list_for_session(
    pool: &SqlitePool,
    session_id: Uuid,
) -> Result<Vec<Message>, AppError> {
    let rows = sqlx::query(
        r#"SELECT id, session_id, role, content, token_count, created_at, metadata
           FROM messages
           WHERE session_id = ?
           ORDER BY created_at ASC"#,
    )
    .bind(session_id.to_string())
    .fetch_all(pool)
    .await?;

    rows.iter().map(row_to_message).collect()
}
