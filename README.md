# co_worker_lite

A Rust backend service that hosts a local LLM (GGUF via [llama.cpp]) and
exposes a chat API over HTTP. SQLite-backed sessions, automatic model
download, OpenAI-shaped endpoints. Foundation for a Claude Code-style
coding agent backend.

## Status

Iteration 1 — request/response only. Streaming, tool calling, auth, and
multi-model serving are intentionally out of scope and will land in later
iterations.

## Requirements

- **Rust** ≥ 1.85 (edition 2024).
- **`cmake`** and a **C++ compiler** — `llama-cpp-2` builds llama.cpp from
  source. On macOS: `brew install cmake` and have Xcode CLT installed.
  On Debian/Ubuntu: `sudo apt install cmake build-essential`.
- **GPU acceleration is opt-in via Cargo features:**
  - `--features metal` on Apple Silicon (recommended on macOS)
  - `--features cuda` on NVIDIA GPUs
  - `--features vulkan` for cross-vendor GPU
  - default build is CPU-only

Disk: the default model (Qwen2.5-Coder-7B-Instruct, Q4_K_M) is ~4.5 GB.

## Build & run

```sh
git clone <repo>
cd co_worker_lite
cp config.toml.example config.toml

# CPU-only:
cargo run --release

# Apple Silicon with Metal:
cargo run --release --features metal

# NVIDIA with CUDA:
cargo run --release --features cuda
```

First launch downloads the model (with a progress bar and `Range`-based
resume). Subsequent launches reuse the cached file in `./models/` and
start in seconds.

## Choosing a model

A curated catalog of coding-focused presets ships with the binary:

```sh
co_worker_lite --list-presets
```

```text
NAME                           SIZE   MIN RAM    CTX  DESCRIPTION
qwen-coder-3b               2.1 GiB     4 GB     8K   Smallest. Snappy on any laptop.
qwen-coder-7b               4.4 GiB     8 GB     8K   Default. Balanced quality / speed.
qwen-coder-14b              8.8 GiB    16 GB    16K   Notably better reasoning; fits a 16 GB Mac.
qwen-coder-32b             19.5 GiB    32 GB    16K   Top coding model in this class.
deepseek-coder-v2-lite     10.4 GiB    16 GB    16K   16B MoE (2.4B active). Fast; very long native ctx.
codestral-22b              13.3 GiB    24 GB    16K   Mistral Codestral. Strong code completion + FIM.
```

Selection precedence (highest first):

1. `--preset <name>` CLI flag
2. `LLM_BACKEND_MODEL_PRESET=<name>` env var
3. `model_preset = "<name>"` in `config.toml`
4. The full `[model]` table in `config.toml` (use this for models that aren't in the catalog)

Examples:

```sh
cargo run --release -- --preset qwen-coder-14b
cargo run --release -- --preset deepseek-coder-v2-lite
LLM_BACKEND_MODEL_PRESET=codestral-22b cargo run --release
```

First run with a new preset downloads the GGUF to `./models/` with resume support; subsequent runs are instant.

**Note on splits.** Some larger HF GGUFs (e.g. Qwen2.5-Coder-32B at higher quants) are split across multiple files. The downloader only handles single-file GGUFs today — every preset above points at a single-file Q4_K_M variant.

## Configuration

Configuration is layered: defaults → `config.toml` → environment
variables prefixed `LLM_BACKEND_`. Nested keys use `__` as the separator.

```sh
LLM_BACKEND_SERVER__PORT=9090 cargo run --release
LLM_BACKEND_MODEL__GPU_LAYERS=0 cargo run --release   # force CPU
```

Logging is controlled by the standard `RUST_LOG` env var, e.g.
`RUST_LOG=info,co_worker_lite=debug`.

See [`config.toml.example`](config.toml.example) for every documented
field.

## API

All endpoints accept and return JSON. Errors come back as
`{"error": {"code": "...", "message": "..."}}` with appropriate HTTP
status codes.

### Health

```sh
curl http://localhost:6969/health
# {"status":"ok","model":"qwen2.5-coder-7b-instruct-q4_k_m.gguf","loaded":true}
```

### List local models

```sh
curl http://localhost:6969/v1/models
```

### Create a session

```sh
curl -X POST http://localhost:6969/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test", "system_prompt": "You are a helpful coding assistant."}'
```

### Send a message

```sh
curl -X POST http://localhost:6969/v1/sessions/<id>/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Write a hello world in Rust", "max_tokens": 512, "temperature": 0.7}'
```

Response shape:

```json
{
  "message": { "id": "...", "role": "assistant", "content": "...", ... },
  "usage": { "prompt_tokens": 42, "completion_tokens": 88, "total_tokens": 130 }
}
```

### List & inspect sessions

```sh
curl 'http://localhost:6969/v1/sessions?limit=20&offset=0'
curl http://localhost:6969/v1/sessions/<id>
curl -X DELETE http://localhost:6969/v1/sessions/<id>
```

## Project layout

```text
co_worker_lite/
├── Cargo.toml
├── config.toml.example
├── claude.md
├── migrations/
│   └── 0001_initial.sql
├── src/
│   ├── main.rs              # init order: config → logging → db → model → server
│   ├── lib.rs               # module roots (also used by integration tests)
│   ├── config.rs            # Settings struct + layered loading
│   ├── error.rs             # AppError + IntoResponse impl
│   ├── state.rs             # AppState (shared via axum::extract::State)
│   ├── types.rs             # request/response types and DB row types
│   ├── model/
│   │   ├── downloader.rs    # HF download with resume + indicatif progress
│   │   ├── engine.rs        # llama-cpp-2 wrapper, InferenceBackend trait
│   │   └── tokenizer.rs     # token counting helpers
│   ├── db/
│   │   ├── sessions.rs      # CRUD for sessions
│   │   └── messages.rs      # CRUD for messages
│   ├── context/             # ContextManager (history fitting)
│   └── api/                 # axum routes
└── tests/
    └── api_test.rs          # 4 integration tests using a stub backend
```

## Tests

```sh
cargo test
```

Integration tests use a `StubBackend` so they don't need to download or
load a real model. To exercise the real engine end-to-end, run with
`--ignored` (currently no `#[ignore]`d tests ship; add your own under
`tests/` if you want one).

## Notes & trade-offs

- **Runtime-checked SQL.** The spec asked for compile-time checked
  queries via `sqlx::query!`, but the macro requires either a live
  `DATABASE_URL` or a checked-in `.sqlx/` offline cache at build time —
  awkward for a fresh checkout. We use `sqlx::query` / `query_as` with
  hand-mapped rows: still type-safe at runtime, and `cargo build` works
  on day one without prep. To migrate later: install `sqlx-cli`, run
  `cargo sqlx prepare`, and switch the call sites.
- **Single in-flight inference.** The engine holds the model behind a
  `Mutex`. Concurrent requests serialize cleanly. Suitable for a
  developer-local backend, not multi-tenant production.
- **Per-request context.** A fresh `LlamaContext` is created per
  request. KV-cache reuse across turns is a future optimization.
- **Token detokenization.** Output text is built one token at a time
  via `token_to_str(.., Special::Plaintext)`; rare multi-byte tokens
  may render imperfectly. Streaming with a proper UTF-8 byte decoder
  is planned for the streaming iteration.

## License

MIT OR Apache-2.0.

[llama.cpp]: https://github.com/ggerganov/llama.cpp
