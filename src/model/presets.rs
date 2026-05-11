//! Curated catalog of model presets.
//!
//! Each preset maps a short name (e.g. `qwen-coder-14b`) to a downloadable
//! GGUF file on Hugging Face. Only **single-file** Q4_K_M variants are
//! included: the downloader does not yet handle multi-part GGUF splits
//! (`*-00001-of-00003.gguf` etc.).
//!
//! Selection precedence at startup (highest first):
//!   1. `--preset <name>` CLI flag
//!   2. `LLM_BACKEND_MODEL_PRESET` env var (via the config crate)
//!   3. `model_preset = "..."` in `config.toml`
//!   4. `[model]` table in `config.toml`
//!   5. Built-in defaults (qwen-coder-7b)

use crate::config::Settings;

#[derive(Debug, Clone, Copy)]
pub struct ModelPreset {
    /// Short name used to select the preset on the CLI / in config.
    pub name: &'static str,
    /// Hugging Face repo id, e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`.
    pub repo: &'static str,
    /// GGUF filename within the repo.
    pub filename: &'static str,
    /// Approximate download size in MiB (for the help listing).
    pub approx_size_mib: u32,
    /// Default context window the engine will use. Caller may shrink this
    /// via config if memory is tight.
    pub context_length: u32,
    /// Native context length of the model itself (informational).
    pub native_context: u32,
    /// Min recommended unified-memory / RAM in GiB to comfortably run the
    /// Q4_K_M variant on Apple Silicon.
    pub min_ram_gib: u32,
    /// One-line summary shown in `--list-presets`.
    pub description: &'static str,
}

pub const PRESETS: &[ModelPreset] = &[
    ModelPreset {
        name: "qwen-coder-3b",
        repo: "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename: "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        approx_size_mib: 2100,
        context_length: 8192,
        native_context: 32768,
        min_ram_gib: 4,
        description: "Smallest. Snappy on any laptop; weakest reasoning.",
    },
    ModelPreset {
        name: "qwen-coder-7b",
        repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        approx_size_mib: 4466,
        context_length: 8192,
        native_context: 32768,
        min_ram_gib: 8,
        description: "Default. Balanced quality / speed for coding tasks.",
    },
    ModelPreset {
        name: "qwen-coder-14b",
        repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename: "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        approx_size_mib: 9011,
        context_length: 16384,
        native_context: 32768,
        min_ram_gib: 16,
        description: "Step up from 7B. Notably better reasoning; fits a 16 GB Mac.",
    },
    ModelPreset {
        name: "qwen-coder-32b",
        repo: "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename: "qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        approx_size_mib: 19927,
        context_length: 16384,
        native_context: 32768,
        min_ram_gib: 32,
        description: "Top coding model in this size class; needs 32 GB+ unified memory.",
    },
    ModelPreset {
        name: "deepseek-coder-v2-lite",
        repo: "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        filename: "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        approx_size_mib: 10649,
        context_length: 16384,
        native_context: 163840,
        min_ram_gib: 16,
        description: "16B MoE (2.4B active). Fast inference, very long native context.",
    },
    ModelPreset {
        name: "codestral-22b",
        repo: "bartowski/Codestral-22B-v0.1-GGUF",
        filename: "Codestral-22B-v0.1-Q4_K_M.gguf",
        approx_size_mib: 13619,
        context_length: 16384,
        native_context: 32768,
        min_ram_gib: 24,
        description: "Mistral Codestral. Strong code completion and fill-in-the-middle.",
    },
];

pub fn find(name: &str) -> Option<&'static ModelPreset> {
    PRESETS.iter().find(|p| p.name.eq_ignore_ascii_case(name))
}

/// Overwrite the model-related fields of `settings` with `preset`'s values.
/// Hardware-dependent fields (`gpu_layers`, `models_dir`) are left alone.
pub fn apply(settings: &mut Settings, preset: &ModelPreset) {
    settings.model.repo = preset.repo.to_string();
    settings.model.filename = preset.filename.to_string();
    // Don't shrink an explicitly-larger context_length the user set in
    // config. Otherwise take the preset's value.
    if settings.model.context_length == ModelDefaults::CONTEXT_LENGTH {
        settings.model.context_length = preset.context_length;
    }
    // sha256 is preset-specific; clear if not set per preset.
    settings.model.sha256.clear();
}

/// Echoes the catalog in a human-friendly aligned table.
pub fn render_listing() -> String {
    let mut out = String::new();
    out.push_str("Available model presets:\n\n");
    out.push_str(&format!(
        "  {:<24}  {:>9}  {:>7}  {:>9}  {}\n",
        "NAME", "SIZE", "MIN RAM", "CTX", "DESCRIPTION"
    ));
    out.push_str(&format!(
        "  {:<24}  {:>9}  {:>7}  {:>9}  {}\n",
        "----", "----", "-------", "---", "-----------"
    ));
    for p in PRESETS {
        let size = if p.approx_size_mib >= 1024 {
            format!("{:.1} GiB", p.approx_size_mib as f32 / 1024.0)
        } else {
            format!("{} MiB", p.approx_size_mib)
        };
        out.push_str(&format!(
            "  {:<24}  {:>9}  {:>5} GB  {:>9}  {}\n",
            p.name,
            size,
            p.min_ram_gib,
            format!("{}K", p.context_length / 1024),
            p.description,
        ));
    }
    out.push_str("\nSelect with: --preset <name>, model_preset = \"<name>\" in config.toml,\nor LLM_BACKEND_MODEL_PRESET=<name>.\n");
    out
}

/// Numeric defaults referenced when deciding whether the user has overridden a
/// field. Keep in sync with `ModelConfig::default()`.
struct ModelDefaults;
impl ModelDefaults {
    const CONTEXT_LENGTH: u32 = 8192;
}
