//! Integration tests for the HTTP API.
//!
//! These tests exercise the full router, database, and context manager but
//! substitute a `StubBackend` for the real LLM so they don't need to download
//! or load a 4.5 GB model. End-to-end tests against the real engine are
//! marked `#[ignore]` and can be opted into with `cargo test -- --ignored`.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use co_worker_lite::api::router;
use co_worker_lite::config::{
    DatabaseConfig, ModelConfig, ServerConfig, Settings,
};
use co_worker_lite::db;
use co_worker_lite::model::engine::{SharedEngine, StubBackend};
use co_worker_lite::state::AppState;
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

struct TestApp {
    app: axum::Router,
    _tempdir: TempDir,
}

async fn build_test_app() -> TestApp {
    let tempdir = TempDir::new().expect("tempdir");
    let db_path = tempdir.path().join("test.db");
    let models_dir = tempdir.path().join("models");
    std::fs::create_dir_all(&models_dir).unwrap();

    let pool = db::init(&db_path).await.expect("db init");
    let engine: SharedEngine = Arc::new(StubBackend::default());
    let settings = Settings {
        models_dir,
        model_preset: None,
        server: ServerConfig::default(),
        model: ModelConfig::default(),
        database: DatabaseConfig {
            path: db_path.clone(),
        },
    };

    let state = AppState {
        settings: Arc::new(settings),
        db: pool,
        engine,
    };
    TestApp {
        app: router(state),
        _tempdir: tempdir,
    }
}

async fn json_body(resp: axum::http::Response<Body>) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .expect("read body");
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let TestApp { app, .. } = build_test_app().await;
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["status"], "ok");
    assert_eq!(body["loaded"], true);
    assert_eq!(body["model"], "stub");
}

#[tokio::test]
async fn create_session_persists_and_returns_id() {
    let TestApp { app, .. } = build_test_app().await;

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"title": "test session", "system_prompt": "be helpful"})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let body = json_body(resp).await;
    let id = body["id"].as_str().expect("id").to_string();
    assert_eq!(body["title"], "test session");
    assert_eq!(body["system_prompt"], "be helpful");

    // Round-trip: GET should return the same session with no messages.
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["id"], id);
    assert_eq!(body["messages"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn message_round_trip_persists_user_and_assistant() {
    let TestApp { app, .. } = build_test_app().await;

    // Create session.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"title": "rt", "system_prompt": null}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let session = json_body(resp).await;
    let id = session["id"].as_str().unwrap().to_string();

    // Send a message.
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/sessions/{id}/messages"))
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"content": "hello there", "max_tokens": 32, "temperature": 0.0})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    let assistant_text = body["message"]["content"].as_str().unwrap();
    assert!(
        assistant_text.contains("hello there"),
        "stub should echo user input, got: {assistant_text}"
    );
    assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);

    // Verify both messages persisted.
    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = json_body(resp).await;
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "hello there");
    assert_eq!(messages[1]["role"], "assistant");
}

#[tokio::test]
async fn delete_session_returns_no_content_and_clears_messages() {
    let TestApp { app, .. } = build_test_app().await;

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"title": "to-delete", "system_prompt": null}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let id = json_body(resp).await["id"].as_str().unwrap().to_string();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);

    let resp = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
