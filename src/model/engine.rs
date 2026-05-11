//! Inference engine.
//!
//! Wraps `llama-cpp-2` behind an `InferenceBackend` trait so handlers and
//! tests can talk to either the real model or a stub. The real implementation
//! holds a single loaded model under a `Mutex`; concurrent requests serialize
//! cleanly through it.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Bound the substring scan when checking stop sequences. We only need to
/// see the tail of the running output since any matching sequence will
/// have its final char(s) at the very end.
const MAX_STOP_TAIL: usize = 128;

use async_trait::async_trait;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::error::AppError;
use crate::types::Role;

/// Type-erased shared inference engine that flows through `AppState`.
pub type SharedEngine = Arc<dyn InferenceBackend>;

/// A single chat turn fed to the engine.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: Role,
    pub content: String,
}

/// Result of a generation call.
#[derive(Debug, Clone)]
pub struct Generated {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Optional parameters for a generation call. Defaults are stable across
/// callers so we don't have to update every site when adding a new knob.
#[derive(Debug, Clone, Default)]
pub struct GenerateOpts {
    /// If any of these substrings appears in the running output, stop
    /// generation immediately and return what we have so far. The matched
    /// stop sequence is included in `Generated.text` — callers strip it
    /// if they care.
    pub stop_sequences: Vec<String>,
}

/// Events produced by `generate_streaming`. The receiver reads tokens as
/// they're sampled and a final `Done` event with the cumulative usage.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
        /// Full generated text (concatenation of all `Token`s). Convenience
        /// so callers don't have to buffer themselves.
        text: String,
    },
    Error(String),
}

#[async_trait]
pub trait InferenceBackend: Send + Sync + 'static {
    fn model_name(&self) -> &str;
    fn context_length(&self) -> u32;
    /// Count tokens in `text` using the model's tokenizer.
    async fn count_tokens(&self, text: &str) -> Result<usize, AppError>;
    /// Run a chat-formatted generation.
    async fn generate(
        &self,
        turns: &[ChatTurn],
        max_tokens: u32,
        temperature: f32,
    ) -> Result<Generated, AppError> {
        self.generate_with(turns, max_tokens, temperature, GenerateOpts::default())
            .await
    }
    /// Same as `generate` but with extra options (e.g. stop sequences for
    /// the agent loop).
    async fn generate_with(
        &self,
        turns: &[ChatTurn],
        max_tokens: u32,
        temperature: f32,
        opts: GenerateOpts,
    ) -> Result<Generated, AppError>;

    /// Stream tokens as they're produced. The default implementation runs
    /// `generate_with` and emits a single `Token` followed by `Done`, so
    /// stubs and remote backends work without a real streaming path.
    async fn generate_streaming(
        &self,
        turns: &[ChatTurn],
        max_tokens: u32,
        temperature: f32,
        opts: GenerateOpts,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>, AppError> {
        let g = self.generate_with(turns, max_tokens, temperature, opts).await?;
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let text = g.text.clone();
        let prompt = g.prompt_tokens;
        let completion = g.completion_tokens;
        tokio::spawn(async move {
            if !text.is_empty() {
                let _ = tx.send(StreamEvent::Token(text.clone())).await;
            }
            let _ = tx
                .send(StreamEvent::Done {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                    text,
                })
                .await;
        });
        Ok(rx)
    }
}

/// Real backend powered by llama.cpp.
pub struct LlamaEngine {
    inner: Arc<Mutex<EngineInner>>,
    model_name: String,
    context_length: u32,
}

struct EngineInner {
    // Backend must outlive model (held below). Both are pinned in this struct.
    _backend: Arc<LlamaBackend>,
    model: LlamaModel,
    template: LlamaChatTemplate,
    n_threads: i32,
}

impl LlamaEngine {
    /// Load a GGUF file. `gpu_layers` semantics match the config:
    /// `-1` = use llama.cpp's default (all layers), `0` = CPU only,
    /// `N > 0` = first N layers on GPU.
    pub fn load(
        backend: Arc<LlamaBackend>,
        model_path: &Path,
        context_length: u32,
        gpu_layers: i32,
    ) -> Result<Self, AppError> {
        let mut model_params = LlamaModelParams::default();
        if gpu_layers >= 0 {
            model_params = model_params.with_n_gpu_layers(gpu_layers as u32);
        }

        tracing::info!(
            path = %model_path.display(),
            gpu_layers,
            context_length,
            "loading model"
        );
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| AppError::Inference(format!("model load failed: {e}")))?;

        // Pull the model's embedded chat template; fall back to ChatML which
        // is what Qwen2.5-Instruct uses anyway.
        let template = match model.chat_template(None) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, "model has no embedded chat template; falling back to chatml");
                LlamaChatTemplate::new("chatml")
                    .map_err(|e| AppError::Inference(format!("chatml template: {e}")))?
            }
        };

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let model_name = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            inner: Arc::new(Mutex::new(EngineInner {
                _backend: backend,
                model,
                template,
                n_threads,
            })),
            model_name,
            context_length,
        })
    }
}

#[async_trait]
impl InferenceBackend for LlamaEngine {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn context_length(&self) -> u32 {
        self.context_length
    }

    async fn count_tokens(&self, text: &str) -> Result<usize, AppError> {
        let inner = self.inner.clone();
        let text = text.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = inner.lock().expect("engine mutex poisoned");
            let toks = guard
                .model
                .str_to_token(&text, AddBos::Never)
                .map_err(|e| AppError::Inference(format!("tokenize: {e}")))?;
            Ok::<usize, AppError>(toks.len())
        })
        .await
        .map_err(|e| AppError::Internal(format!("join error: {e}")))?
    }

    async fn generate_with(
        &self,
        turns: &[ChatTurn],
        max_tokens: u32,
        temperature: f32,
        opts: GenerateOpts,
    ) -> Result<Generated, AppError> {
        let inner = self.inner.clone();
        let context_length = self.context_length;
        let turns: Vec<ChatTurn> = turns.to_vec();
        tokio::task::spawn_blocking(move || -> Result<Generated, AppError> {
            let guard = inner.lock().expect("engine mutex poisoned");
            run_generation(&guard, &turns, max_tokens, temperature, context_length, &opts, None)
        })
        .await
        .map_err(|e| AppError::Internal(format!("join error: {e}")))?
    }

    async fn generate_streaming(
        &self,
        turns: &[ChatTurn],
        max_tokens: u32,
        temperature: f32,
        opts: GenerateOpts,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>, AppError> {
        // Modest buffer — tokens are small, the consumer keeps up.
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamEvent>(64);
        let inner = self.inner.clone();
        let context_length = self.context_length;
        let turns: Vec<ChatTurn> = turns.to_vec();
        tokio::task::spawn_blocking(move || {
            let result: Result<Generated, AppError> = (|| {
                let guard = inner.lock().expect("engine mutex poisoned");
                run_generation(
                    &guard,
                    &turns,
                    max_tokens,
                    temperature,
                    context_length,
                    &opts,
                    Some(&tx),
                )
            })();
            match result {
                Ok(g) => {
                    let _ = tx.blocking_send(StreamEvent::Done {
                        prompt_tokens: g.prompt_tokens,
                        completion_tokens: g.completion_tokens,
                        text: g.text,
                    });
                }
                Err(e) => {
                    let _ = tx.blocking_send(StreamEvent::Error(e.to_string()));
                }
            }
        });
        Ok(rx)
    }
}

// `stream_tx`, when `Some`, receives each generated token piece as it is
// decoded — used by `generate_streaming` to feed an SSE response.
fn run_generation(
    inner: &EngineInner,
    turns: &[ChatTurn],
    max_tokens: u32,
    temperature: f32,
    context_length: u32,
    opts: &GenerateOpts,
    stream_tx: Option<&tokio::sync::mpsc::Sender<StreamEvent>>,
) -> Result<Generated, AppError> {
    let chat: Vec<LlamaChatMessage> = turns
        .iter()
        .map(|t| {
            LlamaChatMessage::new(t.role.as_str().to_string(), t.content.clone())
                .map_err(|e| AppError::Inference(format!("chat message: {e}")))
        })
        .collect::<Result<_, _>>()?;

    let prompt = inner
        .model
        .apply_chat_template(&inner.template, &chat, true)
        .map_err(|e| AppError::Inference(format!("apply chat template: {e}")))?;

    let prompt_tokens = inner
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| AppError::Inference(format!("tokenize prompt: {e}")))?;

    let n_prompt = prompt_tokens.len();
    if n_prompt as u32 >= context_length {
        return Err(AppError::BadRequest(format!(
            "prompt of {n_prompt} tokens exceeds context length {context_length}"
        )));
    }

    let n_ctx = NonZeroU32::new(context_length)
        .ok_or_else(|| AppError::Internal("context_length must be > 0".into()))?;
    // n_batch is the most tokens llama.cpp will accept in a single
    // `decode()` call. Keep it modest (memory) and feed the prompt in
    // chunks below. Capped at the context length for tiny configs.
    let n_batch: u32 = std::cmp::min(10000, context_length);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch)
        .with_n_threads(inner.n_threads)
        .with_n_threads_batch(inner.n_threads);

    let mut ctx = inner
        .model
        .new_context(&inner._backend, ctx_params)
        .map_err(|e| AppError::Inference(format!("create context: {e}")))?;

    let mut batch = LlamaBatch::new(n_batch as usize, 1);

    // Feed the prompt to the context in chunks of at most `n_batch` tokens.
    // Only the very last token of the prompt needs its logits — every
    // earlier token is just KV-cache fill.
    let last_idx = n_prompt - 1;
    let mut chunk_start = 0usize;
    while chunk_start < n_prompt {
        let chunk_end = std::cmp::min(chunk_start + n_batch as usize, n_prompt);
        batch.clear();
        for i in chunk_start..chunk_end {
            batch
                .add(prompt_tokens[i], i as i32, &[0], i == last_idx)
                .map_err(|e| AppError::Inference(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| AppError::Inference(format!("decode prompt: {e}")))?;
        chunk_start = chunk_end;
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(42);

    let mut sampler = if temperature <= 0.0 {
        LlamaSampler::chain_simple([LlamaSampler::greedy()])
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::temp(temperature),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::dist(seed),
        ])
    };

    let eos = inner.model.token_eos();
    let mut output = String::new();
    let mut completion_tokens = 0u32;
    let mut pos = n_prompt as i32;
    let max_pos = context_length as i32;
    // Streaming UTF-8 decoder so tokens whose bytes split a multi-byte
    // codepoint render correctly across iterations.
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let mut hit_stop = false;
    while completion_tokens < max_tokens && pos < max_pos {
        let next = sampler.sample(&ctx, -1);
        sampler.accept(next);

        if next == eos {
            break;
        }

        if let Ok(piece) = inner.model.token_to_piece(next, &mut decoder, false, None) {
            if !piece.is_empty() {
                if let Some(tx) = stream_tx {
                    // Best-effort send; the receiver may have dropped.
                    let _ = tx.blocking_send(StreamEvent::Token(piece.clone()));
                }
                output.push_str(&piece);
            }
        }
        completion_tokens += 1;

        // Cheap O(n*k) substring check; stop sequences are short and few.
        // We check only the tail to keep the work bounded.
        if !opts.stop_sequences.is_empty() {
            let tail_start = output.len().saturating_sub(MAX_STOP_TAIL);
            let tail = &output[tail_start..];
            if opts.stop_sequences.iter().any(|s| tail.contains(s.as_str())) {
                hit_stop = true;
                break;
            }
        }

        batch.clear();
        batch
            .add(next, pos, &[0], true)
            .map_err(|e| AppError::Inference(format!("batch add: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| AppError::Inference(format!("decode token: {e}")))?;
        pos += 1;
    }
    let _ = hit_stop;

    Ok(Generated {
        text: output,
        prompt_tokens: n_prompt as u32,
        completion_tokens,
    })
}

/// In-memory stub backend for tests. Echoes a deterministic response and
/// counts tokens by whitespace-splitting (good enough for accounting tests).
pub struct StubBackend {
    pub model_name: String,
    pub context_length: u32,
}

impl Default for StubBackend {
    fn default() -> Self {
        Self {
            model_name: "stub".into(),
            context_length: 4096,
        }
    }
}

#[async_trait]
impl InferenceBackend for StubBackend {
    fn model_name(&self) -> &str {
        &self.model_name
    }
    fn context_length(&self) -> u32 {
        self.context_length
    }
    async fn count_tokens(&self, text: &str) -> Result<usize, AppError> {
        Ok(text.split_whitespace().count().max(1))
    }
    async fn generate_with(
        &self,
        turns: &[ChatTurn],
        _max_tokens: u32,
        _temperature: f32,
        _opts: GenerateOpts,
    ) -> Result<Generated, AppError> {
        let last_user = turns
            .iter()
            .rev()
            .find(|t| t.role == Role::User)
            .map(|t| t.content.as_str())
            .unwrap_or("");
        let text = format!("[stub] echo: {last_user}");
        let prompt_tokens = turns
            .iter()
            .map(|t| t.content.split_whitespace().count() as u32)
            .sum();
        let completion_tokens = text.split_whitespace().count() as u32;
        Ok(Generated {
            text,
            prompt_tokens,
            completion_tokens,
        })
    }
}
