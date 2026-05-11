# co_worker_lite

Rust backend service that hosts a local LLM (GGUF via [llama.cpp]) and exposes a chat API. Foundation for a Claude Code-style coding agent backend. User-facing docs live in [README.md](README.md); this file is the engineering map.

## Iteration scope

Iteration 1 is request/response only. Streaming, tool calling, auth, and multi-model serving are deliberately out of scope and slated for later.

## Build, run, test

```sh
brew install cmake             # one-time; needed by llama-cpp-2 to build llama.cpp
cargo build                    # CPU-only default
cargo build --features metal   # Apple Silicon
cargo build --features cuda    # NVIDIA
cargo test                     # 2 unit + 4 integration tests, all hermetic (StubBackend)
cargo run --release            # downloads ~4.5 GB on first launch
```

`config.toml` is loaded if present, then overlaid with `LLM_BACKEND_*__*` env vars. See [config.toml.example](config.toml.example).

## Layout

- [src/main.rs](src/main.rs) — init order: config → tracing → DB → model download → model load → axum server. Handles SIGINT/SIGTERM for graceful shutdown.
- [src/lib.rs](src/lib.rs) — module roots; re-exported so `tests/` and `main.rs` share the same surface.
- [src/config.rs](src/config.rs) — `Settings`, layered loader (defaults → file → env).
- [src/error.rs](src/error.rs) — `AppError` enum (`thiserror`) with `IntoResponse` so handlers return `Result<T, AppError>` and get uniform JSON error bodies.
- [src/state.rs](src/state.rs) — `AppState { settings, db, engine }`.
- [src/types.rs](src/types.rs) — request/response and DB row types. Roughly OpenAI-shaped.
- [src/db/](src/db/) — SQLite via sqlx. [mod.rs](src/db/mod.rs) opens the pool and runs migrations; [sessions.rs](src/db/sessions.rs) and [messages.rs](src/db/messages.rs) are CRUD.
- [src/model/](src/model/) — [downloader.rs](src/model/downloader.rs) (HF download with `Range` resume + indicatif progress + optional SHA256), [engine.rs](src/model/engine.rs) (`InferenceBackend` trait + `LlamaEngine` impl + `StubBackend` for tests), [tokenizer.rs](src/model/tokenizer.rs), [presets.rs](src/model/presets.rs) (curated catalog of model presets surfaced via `--list-presets`).
- [src/context/mod.rs](src/context/mod.rs) — `ContextManager`. Loads history, recomputes missing token counts, drops oldest user/assistant pairs first, never splits a turn, always preserves system prompt + trailing user message.
- [src/api/](src/api/) — axum router. [health.rs](src/api/health.rs) `/health`, [models.rs](src/api/models.rs) `/v1/models`, [sessions.rs](src/api/sessions.rs) `/v1/sessions[/:id]`, [messages.rs](src/api/messages.rs) `POST /v1/sessions/:id/messages`.
- [migrations/0001_initial.sql](migrations/0001_initial.sql) — `sessions` + `messages` tables, FK + indices.
- [tests/api_test.rs](tests/api_test.rs) — 4 hermetic integration tests using `tower::ServiceExt::oneshot` against the assembled router with a `StubBackend`.

## Architecture conventions

- **Inference is single-flight.** `LlamaEngine` holds the model under `Arc<Mutex<…>>`; concurrent requests serialize through it. Each `generate()` call hops to `tokio::task::spawn_blocking` so the async runtime stays free.
- **`InferenceBackend` is a trait, not a concrete type.** Tests use `StubBackend`; production uses `LlamaEngine`. `SharedEngine = Arc<dyn InferenceBackend>`. New backends (mistral.rs, candle, remote API) plug in by implementing this trait.
- **Per-request `LlamaContext`.** The model stays loaded; the context is recreated per-call. KV-cache reuse across turns is a future optimization.
- **GPU acceleration is opt-in via Cargo features.** `metal`, `cuda`, `vulkan` map to the corresponding `llama-cpp-2` features. Default build is CPU-only. Runtime `model.gpu_layers` controls how many layers are offloaded.
- **Errors flow through one type.** Anywhere user code returns `Result`, it should return `Result<_, AppError>`. The `IntoResponse` impl maps variants → HTTP status. `anyhow` is used **only** at the top level in `main.rs`.

## Trade-offs worth knowing

- **Runtime-checked SQL, not compile-time.** The original spec asked for `sqlx::query!`, but that needs a build-time `DATABASE_URL` or a checked-in `.sqlx/` cache. We use `sqlx::query` / `query_as` with hand-written `FromRow`-style mapping so `cargo build` works on a fresh checkout. To migrate later: install `sqlx-cli`, run `cargo sqlx prepare`, switch the call sites.
- **DB pool warm-up.** [src/db/mod.rs](src/db/mod.rs) sets `min_connections(1)` and runs `SELECT 1` after `migrate!`. Without this, the first handler query against a freshly-initialized SQLite pool intermittently fails with `SQLITE_CANTOPEN` (14). The warm-up surfaces any init failure during startup instead of on the first request.
- **Detokenization uses `token_to_piece` with a streaming `encoding_rs::UTF_8` decoder.** Multi-byte codepoints split across tokens render correctly. Avoid the deprecated `token_to_str` / `Special` API.
- **Reqwest, not hf-hub.** The spec allowed either. Plain reqwest gives us cleaner integration with `indicatif` progress and `Range`-based resume. Trade: no built-in HF auth — fine for public Qwen models.

## Testing strategy

- Unit tests for the trim policy live alongside [src/context/mod.rs](src/context/mod.rs).
- Integration tests in [tests/api_test.rs](tests/api_test.rs) build the real router with a real SQLite tempfile but a `StubBackend`. They exercise health, session create/list/get/delete, and a full message round-trip.
- Real-engine smoke tests are not shipped; if you add one, gate with `#[ignore]` so CI/`cargo test` stays fast.

## Model presets

[src/model/presets.rs](src/model/presets.rs) ships a small, curated list of single-file Q4_K_M GGUFs. Each preset is a `&'static ModelPreset` with `repo`, `filename`, `context_length`, an RAM hint, and a description.

Selection precedence at startup (highest first):

1. `--preset <name>` CLI flag (parsed in [src/main.rs](src/main.rs))
2. `LLM_BACKEND_MODEL_PRESET=<name>` env var (layered in via the `config` crate)
3. `model_preset = "..."` in `config.toml` (deserialized into `Settings::model_preset`)
4. The full `[model]` table

When a preset is selected, `presets::apply` overwrites `settings.model.{repo, filename, sha256}` and bumps `context_length` only if the user has not overridden it (heuristic: it still matches `ModelDefaults::CONTEXT_LENGTH`). Hardware-dependent fields (`gpu_layers`, `models_dir`) are left untouched.

**Adding a preset:** confirm the GGUF exists as a *single* file in the HF repo (`/tree/main`). Multi-part splits (`*-00001-of-00003.gguf`) aren't supported by the downloader yet — they'd need range-resume across part boundaries plus llama.cpp's split-file loader. Then append a `ModelPreset { ... }` literal to `PRESETS`.

## When making changes

- New endpoint → add a module under [src/api/](src/api/), wire into [src/api/mod.rs](src/api/mod.rs).
- New DB table → add a numbered file under [migrations/](migrations/), add a sibling module under [src/db/](src/db/), update [src/types.rs](src/types.rs).
- New backend (e.g., remote API mode) → impl `InferenceBackend` in [src/model/engine.rs](src/model/engine.rs) (or a new module); pick at startup in [src/main.rs](src/main.rs).
- Touching token accounting → re-check [src/context/mod.rs](src/context/mod.rs) trim policy and the unit tests there.
- Bumping `llama-cpp-2` → its API moves between minor versions; expect to fix call sites in [src/model/engine.rs](src/model/engine.rs).

[llama.cpp]: https://github.com/ggerganov/llama.cpp
