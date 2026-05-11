//! HTTP API surface.
//!
//! Builds the axum router and mounts module-specific handlers. The router
//! is roughly OpenAI-compatible in shape (sessions ≈ conversations,
//! messages ≈ chat completions) so a generic client can later target this
//! backend or OpenAI through a thin adapter.

use std::time::Duration;

use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub mod agent;
pub mod chat;
pub mod health;
pub mod memories;
pub mod messages;
pub mod models;
pub mod sessions;

pub fn router(state: AppState) -> Router {
    Router::new()
        .merge(health::router())
        .merge(models::router())
        .merge(sessions::router())
        .merge(messages::router())
        .merge(memories::router())
        .merge(chat::router())
        .merge(agent::router())
        .with_state(state)
        .layer(CorsLayer::permissive())
        .layer(
            TraceLayer::new_for_http()
                .on_request(())
                .on_response(
                    |response: &axum::http::Response<_>, latency: Duration, _: &tracing::Span| {
                        tracing::info!(
                            status = response.status().as_u16(),
                            latency_ms = latency.as_millis() as u64,
                            "request"
                        );
                    },
                ),
        )
}
