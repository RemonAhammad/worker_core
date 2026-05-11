//! Shared application state passed to every handler via `axum::extract::State`.

use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use sqlx::SqlitePool;
use tokio::sync::RwLock;

use crate::config::Settings;
use crate::model::engine::SharedEngine;

/// Runtime-swappable inference engine.
///
/// Hot-swapping the loaded GGUF (when the user picks a different model in
/// the UI) needs to replace `state.engine` while in-flight handlers may
/// still hold a reference. We store the engine inside a `tokio::sync::RwLock`
/// and hand out cloned `Arc<dyn InferenceBackend>`s on read so concurrent
/// requests can keep working on the *old* engine until they finish, while
/// new requests pick up the new one. Memory of the old engine is freed
/// when the last clone is dropped.
pub struct EngineSlot {
    inner: RwLock<SharedEngine>,
}

impl EngineSlot {
    pub fn new(engine: SharedEngine) -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(engine),
        })
    }

    /// Snapshot the currently-active engine. Cheap (`Arc::clone`).
    pub async fn current(&self) -> SharedEngine {
        self.inner.read().await.clone()
    }

    /// Replace the active engine atomically. Returns the old one so the
    /// caller can decide when to drop it.
    pub async fn replace(&self, new: SharedEngine) -> SharedEngine {
        let mut g = self.inner.write().await;
        std::mem::replace(&mut *g, new)
    }
}

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub db: SqlitePool,
    pub engine: Arc<EngineSlot>,
    /// Kept around so the `/v1/models/load` handler can spin up a new
    /// `LlamaEngine` without re-initializing the global backend (which
    /// llama.cpp only allows once per process). `None` in test mode.
    pub llama_backend: Option<Arc<LlamaBackend>>,
}
