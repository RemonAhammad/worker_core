# co_worker_lite

Rust backend service that hosts a local LLM (GGUF via [llama.cpp]) and exposes a chat **and agent** API. Foundation for a Claude Code-style coding agent backend. User-facing docs live in [README.md](README.md); this file is the engineering map.

## Iteration scope

- **Iteration 1 (shipped)** — request/response chat, sessions, memories, sticky `/v1/chat`, multi-model with hot-swap, per-session model pin.
- **Iteration 2 (shipped)** — agentic loop with filesystem tool calling, stop-sequence-driven inference, token streaming over SSE.
- Out of scope still: auth, multi-tenant safety, image/audio modality.

## Build, run, test

```sh
brew install cmake             # one-time; needed by llama-cpp-2 to build llama.cpp
cargo build                    # CPU-only default
cargo build --features metal   # Apple Silicon
cargo build --features cuda    # NVIDIA
cargo test                     # 4 integration + 16 unit tests, all hermetic (StubBackend)
cargo run --release            # downloads ~4.5 GB on first launch
cargo run --release -- --preset qwen-coder-14b   # different model
cargo run --release -- --list-presets            # show the catalog
```

`config.toml` is loaded if present, then overlaid with `LLM_BACKEND_*__*` env vars. See [config.toml.example](config.toml.example).

## Layout

- [src/main.rs](src/main.rs) — init order: clap → config → tracing → DB → model download → model load → axum server. Builds `EngineSlot` so models can be hot-swapped without restarting. Handles SIGINT/SIGTERM for graceful shutdown.
- [src/lib.rs](src/lib.rs) — module roots; re-exported so `tests/` and `main.rs` share the same surface.
- [src/config.rs](src/config.rs) — `Settings`, layered loader (defaults → file → env), optional `model_preset` field.
- [src/error.rs](src/error.rs) — `AppError` enum (`thiserror`) with `IntoResponse` so handlers return `Result<T, AppError>` and get uniform JSON error bodies.
- [src/state.rs](src/state.rs) — `AppState { settings, db, engine: Arc<EngineSlot>, llama_backend: Option<Arc<LlamaBackend>> }`. `EngineSlot` wraps `SharedEngine` in a `tokio::sync::RwLock` so handlers snapshot the engine per-request while `/v1/models/load` swaps atomically. `llama_backend` is the cached global init handle reused by every load (llama.cpp only allows init once per process).
- [src/types.rs](src/types.rs) — request/response and DB row types. Roughly OpenAI-shaped.
- [src/db/](src/db/) — SQLite via sqlx. [mod.rs](src/db/mod.rs) opens the pool and runs migrations; [sessions.rs](src/db/sessions.rs) (`create`, `get`, `list`, `delete`, `update` for title/system_prompt/model_name, `touch`, `most_recent`), [messages.rs](src/db/messages.rs), [memories.rs](src/db/memories.rs).
- [src/model/](src/model/) — [downloader.rs](src/model/downloader.rs) (HF download with `Range` resume + indicatif progress + optional SHA256), [engine.rs](src/model/engine.rs) (`InferenceBackend` trait + `LlamaEngine` impl + `StubBackend` + `GenerateOpts` for stop sequences + `generate_streaming` token channel), [tokenizer.rs](src/model/tokenizer.rs), [presets.rs](src/model/presets.rs) (curated catalog of model presets surfaced via `--list-presets`).
- [src/context/mod.rs](src/context/mod.rs) — `ContextManager`. Loads history, recomputes missing token counts, drops oldest user/assistant pairs first, never splits a turn, always preserves system prompt + trailing user message. Injects long-term memories at the top of the system block.
- [src/memory/mod.rs](src/memory/mod.rs) — regex-based auto-extractor for self-introduction facts ("my name is X", "i live in Y") run on every user turn.
- [src/tools/](src/tools/) — agent tool catalog and parser:
  - [mod.rs](src/tools/mod.rs) — `filesystem_tools()` returns the 8 GGML-style `ToolDefinition`s the model sees inside the `<tools>` system block. `render_tool_preamble` builds the system prompt addendum. `render_tool_response` formats a `ToolResult` back to the model as a `<tool_response>` block.
  - [parser.rs](src/tools/parser.rs) — extracts `<tool_call>{"name":..,"arguments":..}</tool_call>` blocks from model output, tolerating codefence wrappers and unterminated blocks. Returns the prose tail separately so only user-visible text is persisted. 6 unit tests.
- [src/api/](src/api/) — axum router (CORS permissive, request tracing layer):
  - [health.rs](src/api/health.rs) `GET /health`
  - [models.rs](src/api/models.rs) `GET /v1/models` (legacy flat list), `GET /v1/models/catalog` (presets + local merged), `POST /v1/models/load` (hot-swap with download). Exports `ensure_model_loaded()` used by chat/agent/messages handlers to honor per-session model pins.
  - [sessions.rs](src/api/sessions.rs) `POST/GET /v1/sessions`, `GET/DELETE/PATCH /v1/sessions/:id`, `GET /v1/sessions/:id/debug`. PATCH accepts `title`, `system_prompt`, `model_name`.
  - [messages.rs](src/api/messages.rs) `POST /v1/sessions/:id/messages` (request/response) and `POST /v1/sessions/:id/messages/stream` (Server-Sent Events).
  - [chat.rs](src/api/chat.rs) `POST /v1/chat` — sticky session.
  - [memories.rs](src/api/memories.rs) `/v1/memories[/:id]`.
  - [agent.rs](src/api/agent.rs) — agentic loop endpoints `POST /v1/sessions/:id/agent` and `POST /v1/sessions/:id/agent/continue`. Returns a discriminated `AgentResponse::Message` or `AgentResponse::ToolCalls`. Uses `</tool_call>` as a stop sequence so generation halts at the first call instead of running to EOS.
- [migrations/0001_initial.sql](migrations/0001_initial.sql) — `sessions` + `messages` tables. [migrations/0002_memories.sql](migrations/0002_memories.sql) — `memories` table.
- [tests/api_test.rs](tests/api_test.rs) — 4 hermetic integration tests using `tower::ServiceExt::oneshot` against the assembled router with a `StubBackend`.

## Architecture conventions

- **Inference is single-flight at the engine level.** `LlamaEngine` holds the model under `Arc<Mutex<…>>`; concurrent requests serialize through it. Each `generate*` call hops to `tokio::task::spawn_blocking` so the async runtime stays free.
- **Engine is hot-swappable at the state level.** `EngineSlot` wraps `Arc<dyn InferenceBackend>` in a `tokio::sync::RwLock`. `state.engine.current().await` returns a snapshot every handler uses for the duration of one request — a mid-request `/v1/models/load` only affects the *next* request.
- **`InferenceBackend` has three methods**: `count_tokens`, `generate_with(opts)`, `generate_streaming(opts)`. The trait provides a default `generate_streaming` impl that wraps `generate_with` (works for `StubBackend` and any future remote backend). `LlamaEngine` overrides both with a real token-streaming impl.
- **Agent loop persists state in messages.** `<tool_call>` markup is saved verbatim on the assistant message; tool results land as `Role::Tool` messages with `<tool_response>` payloads. History trimming, memory injection, and session resume keep working unchanged.
- **Per-session model pinning is lazy.** `sessions.model_name` is the authoritative pin. Chat/messages/agent handlers call `models::ensure_model_loaded(&state, &session.model_name)` before snapshotting the engine; if it differs from currently loaded, that request triggers a full download-and-swap.
- **GPU acceleration is opt-in via Cargo features.** `metal`, `cuda`, `vulkan` map to the corresponding `llama-cpp-2` features. Default build is CPU-only. Runtime `model.gpu_layers` controls how many layers are offloaded.
- **Errors flow through one type.** Anywhere user code returns `Result`, it should return `Result<_, AppError>`. The `IntoResponse` impl maps variants → HTTP status. `anyhow` is used **only** at the top level in `main.rs`.

## Streaming pipeline

`POST /v1/sessions/:id/messages/stream` is the user-facing endpoint. Internals:

1. Handler builds the prompt as usual (session pinned model load, snapshot, ContextManager).
2. Calls `engine.generate_streaming(...)` which returns a `tokio::sync::mpsc::Receiver<StreamEvent>`.
3. `LlamaEngine::generate_streaming` spawns a blocking task that runs the standard generation loop, but calls `tx.blocking_send(StreamEvent::Token(piece))` after every token. When the loop finishes naturally it sends `StreamEvent::Done { prompt_tokens, completion_tokens, text }`. Errors become `StreamEvent::Error(msg)`.
4. The handler wraps the receiver in `ReceiverStream`, maps each event to an SSE `Event`, and uses `async_stream::stream!` to splice in a final DB persist + `done` event with the saved row.

Frontend consumes via plain `fetch` against the SSE endpoint (no plugin command — the desktop reads `getBaseUrl()` and hits the backend directly). See [`runStreamingChat`](../co_worker_cli/apps/desktop/src/lib/stores.ts) on the desktop side.

The agent endpoint is **not** streamed; the loop structure makes it lower-value (tool calls land at the end of each turn anyway).

## Trade-offs worth knowing

- **Runtime-checked SQL, not compile-time.** `sqlx::query` / `query_as` with hand-written `FromRow`-style mapping so `cargo build` works on a fresh checkout without `DATABASE_URL` or a `.sqlx/` cache.
- **DB pool warm-up.** [src/db/mod.rs](src/db/mod.rs) sets `min_connections(1)` and runs `SELECT 1` after `migrate!`. Without this, the first handler query against a freshly-initialized SQLite pool intermittently fails with `SQLITE_CANTOPEN` (14).
- **Detokenization uses `token_to_piece` with a streaming `encoding_rs::UTF_8` decoder.** Multi-byte codepoints split across tokens render correctly.
- **Reqwest, not hf-hub, for downloads.** Cleaner integration with `indicatif` progress and `Range`-based resume; trade is no built-in HF auth.
- **Stop-sequence implementation is a tail substring scan.** Each generated piece appends to the running output; we check the last 128 chars for any stop. Good enough for short sentinels (`</tool_call>`); not a generic substring engine.
- **Agent token streaming is not exposed yet.** The infra exists (`generate_streaming` works) but the agent endpoint's split-output / persist-tool-call logic depends on the *complete* model output. Streaming the agent loop is a future iteration.

## Testing strategy

- Unit tests for the trim policy live alongside [src/context/mod.rs](src/context/mod.rs).
- Unit tests for tool-call parsing live in [src/tools/parser.rs](src/tools/parser.rs) — 6 cases covering single/multi calls, malformed bodies, unterminated blocks, codefence-wrapped JSON.
- Unit tests for memory extraction in [src/memory/](src/memory/).
- Integration tests in [tests/api_test.rs](tests/api_test.rs) build the real router with a real SQLite tempfile but a `StubBackend`. They exercise health, session create/list/get/delete, and a full message round-trip.
- Real-engine smoke tests are not shipped; if you add one, gate with `#[ignore]` so CI/`cargo test` stays fast.

## Model presets

[src/model/presets.rs](src/model/presets.rs) ships a small, curated list of single-file Q4_K_M GGUFs. Each preset is a `&'static ModelPreset` with `repo`, `filename`, `context_length`, an RAM hint, and a description. Selection precedence at startup: `--preset` CLI flag → `LLM_BACKEND_MODEL_PRESET` env → `model_preset` in config.toml → the full `[model]` table.

## When making changes

- **New endpoint** → add a module under [src/api/](src/api/), wire into [src/api/mod.rs](src/api/mod.rs).
- **New DB table** → add a numbered file under [migrations/](migrations/), add a sibling module under [src/db/](src/db/), update [src/types.rs](src/types.rs).
- **New backend** (e.g., remote API mode) → impl `InferenceBackend` in [src/model/engine.rs](src/model/engine.rs) (or a new module); pick at startup in [src/main.rs](src/main.rs). The default `generate_streaming` impl means you only need `generate_with` and `count_tokens`.
- **New tool** the agent can call → add to `filesystem_tools()` in [src/tools/mod.rs](src/tools/mod.rs) (declaration only — the backend never executes tools, it just declares them). Implement on the client side (`tauri-plugin-co-worker/src/tools.rs`) and dispatch in the desktop's `runTool` switch.
- **Touching token accounting** → re-check [src/context/mod.rs](src/context/mod.rs) trim policy and the unit tests there.
- **Bumping `llama-cpp-2`** → its API moves between minor versions; expect to fix call sites in [src/model/engine.rs](src/model/engine.rs).
- **Touching the agent loop** → the prose vs. `<tool_call>` split lives in [src/tools/parser.rs](src/tools/parser.rs); the system-prompt augmentation in [src/tools/mod.rs](src/tools/mod.rs); the persist-and-decide flow in [src/api/agent.rs](src/api/agent.rs).

[llama.cpp]: https://github.com/ggerganov/llama.cpp
