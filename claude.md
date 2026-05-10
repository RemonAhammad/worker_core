# co_worker_lite

A Rust project, currently in bootstrap state.

## Status

Fresh scaffold. No commits yet. `src/main.rs` is the default `cargo new` hello-world; no application logic has been written.

## Layout

- [Cargo.toml](Cargo.toml) — package manifest. Edition 2024, version 0.1.0, no dependencies declared yet.
- [src/main.rs](src/main.rs) — binary entry point (currently a stub).
- [models/](models/) — gitignored directory, present but empty. Intended to hold model artifacts (weights, GGUF/ONNX, etc.) that should not be checked in.
- [.gitignore](.gitignore) — ignores `/target`, `*.lock`, and `/models`.

Note: `*.lock` is gitignored, so `Cargo.lock` is not tracked. For a binary crate this is unusual — the Cargo convention is to commit `Cargo.lock` for reproducible builds. Worth revisiting once dependencies are added.

## Build & run

```
cargo build
cargo run
```

Toolchain: Rust edition 2024 (requires a recent stable Rust — 1.85+).

## Conventions

None established yet. Update this file as architecture, module boundaries, or workflows take shape.
