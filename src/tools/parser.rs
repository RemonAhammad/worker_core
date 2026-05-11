//! Extract tool calls from model output, tolerant of two formats:
//!
//! 1. **Canonical** — `<tool_call>{"name":..,"arguments":..}</tool_call>`,
//!    the Qwen2.5-Coder Hermes-style format we ask for in the preamble.
//!
//! 2. **Fallback** — a ```json fenced code block whose body parses as
//!    `{"name":..,"arguments":..}`. Weaker chat models (including the 7B
//!    variant of Qwen2.5-Coder) sometimes ignore our preamble and emit the
//!    function call this way; accepting it lets the agent loop succeed
//!    instead of returning the JSON to the user as prose.
//!
//! When both wrappers are present we prefer the canonical one and let the
//! fenced fallback fill in only when no `<tool_call>` block exists.
//!
//! The function also returns the "prose tail" of the output — everything
//! outside the consumed tool-call regions — so the assistant message saved
//! to the DB only contains text the user should see.

use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use super::ParsedToolCall;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FENCE_JSON: &str = "```json";
const FENCE: &str = "```";

#[derive(Debug, Deserialize)]
struct ToolCallBody {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// Result of scanning a model output for tool calls.
#[derive(Debug, Default)]
pub struct ParseOutcome {
    /// Prose text the user should see. Everything BEFORE the first tool
    /// call survives; the section spanning the first `<tool_call>` to the
    /// last `</tool_call>` is removed.
    pub prose: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

pub fn parse(output: &str) -> ParseOutcome {
    let mut tool_calls = Vec::new();
    let mut prose = String::new();
    let mut cursor = 0usize;
    let mut found_xml = false;

    // Pass 1: canonical `<tool_call>...</tool_call>` blocks.
    while let Some(rel) = output[cursor..].find(OPEN) {
        found_xml = true;
        let open_idx = cursor + rel;
        prose.push_str(&output[cursor..open_idx]);
        let body_start = open_idx + OPEN.len();
        let Some(close_rel) = output[body_start..].find(CLOSE) else {
            // Unterminated block — bail. Treat the rest as prose so we
            // don't lose user-visible text.
            prose.push_str(&output[open_idx..]);
            cursor = output.len();
            break;
        };
        let body_end = body_start + close_rel;
        let body = output[body_start..body_end].trim();

        match parse_one_body(body) {
            Some(call) => tool_calls.push(call),
            None => {
                tracing::warn!(body = body, "could not parse tool_call body — leaving as prose");
                prose.push_str(&output[open_idx..body_end + CLOSE.len()]);
            }
        }
        cursor = body_end + CLOSE.len();
    }
    prose.push_str(&output[cursor..]);

    // Pass 2 (fallback): if the model didn't use `<tool_call>` tags AT ALL,
    // scan the output for ```json fenced blocks and treat any body that
    // shape-matches a tool-call payload as one. Without this, weaker
    // models that emit raw JSON inside code fences would have their tool
    // calls rendered as prose to the user, which is the failure mode the
    // user actually hit on Qwen2.5-Coder-7B.
    if !found_xml {
        let (extra_calls, fallback_prose) = parse_fenced_fallback(&prose);
        if !extra_calls.is_empty() {
            tracing::debug!(
                n = extra_calls.len(),
                "parser: accepted fenced JSON tool calls (no <tool_call> tags)"
            );
            tool_calls.extend(extra_calls);
            prose = fallback_prose;
        }
    }

    ParseOutcome {
        prose: prose.trim().to_string(),
        tool_calls,
    }
}

/// Scan `text` for ```json ... ``` blocks. For each block whose body is a
/// JSON object with `name` (and ideally `arguments`) fields, treat it as a
/// tool call and remove it from the prose.
fn parse_fenced_fallback(text: &str) -> (Vec<ParsedToolCall>, String) {
    let mut calls = Vec::new();
    let mut prose = String::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(FENCE) {
        let open_idx = cursor + rel;
        prose.push_str(&text[cursor..open_idx]);

        // Determine fence opening token (```json or just ```) and start of body.
        let after_fence = open_idx + FENCE.len();
        let (body_start, _had_json_label) =
            if text[open_idx..].starts_with(FENCE_JSON) {
                (open_idx + FENCE_JSON.len(), true)
            } else {
                (after_fence, false)
            };
        // Skip optional newline.
        let body_start = if text.as_bytes().get(body_start).copied() == Some(b'\n') {
            body_start + 1
        } else {
            body_start
        };

        // Look for the closing fence.
        let Some(close_rel) = text[body_start..].find(FENCE) else {
            // Unterminated fence — preserve and stop.
            prose.push_str(&text[open_idx..]);
            cursor = text.len();
            break;
        };
        let body_end = body_start + close_rel;
        let body = text[body_start..body_end].trim();
        let close_end = body_end + FENCE.len();

        let extracted = parse_bodies(body);
        if extracted.is_empty() {
            // Not a tool call — keep the fence intact in the prose.
            prose.push_str(&text[open_idx..close_end]);
        } else {
            calls.extend(extracted);
        }
        cursor = close_end;
    }
    prose.push_str(&text[cursor..]);
    (calls, prose)
}

/// Parse one or more tool-call objects from a `body` string. Returns a Vec
/// so callers can handle: a single object, a JSON array of objects, or
/// multiple newline-delimited objects concatenated in the same block —
/// the last variant being what Qwen2.5-Coder-7B emits when it wants to
/// invoke several tools in one turn.
fn parse_bodies(body: &str) -> Vec<ParsedToolCall> {
    let stripped = body
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if stripped.is_empty() {
        return Vec::new();
    }

    // Try the easy cases first: a single object, then a JSON array.
    if let Ok(parsed) = serde_json::from_str::<ToolCallBody>(stripped) {
        return vec![make_call(parsed)];
    }
    if let Ok(arr) = serde_json::from_str::<Vec<Value>>(stripped) {
        return arr
            .into_iter()
            .filter_map(|v| serde_json::from_value::<ToolCallBody>(v).ok())
            .map(make_call)
            .collect();
    }

    // Hardest case: multiple JSON objects concatenated, possibly separated
    // by newlines, commas, or nothing. `serde_json::Deserializer::from_str`
    // exposes a stream iterator that walks one value at a time.
    let de = serde_json::Deserializer::from_str(stripped);
    let mut out = Vec::new();
    for v in de.into_iter::<Value>().flatten() {
        if let Ok(call) = serde_json::from_value::<ToolCallBody>(v) {
            out.push(make_call(call));
        }
    }
    out
}

/// Compat shim for the canonical `<tool_call>` branch which still treats
/// each block as one call. We keep its old "return Option" contract by
/// collapsing a multi-object body into the first call, on the principle
/// that the XML wrapper is supposed to hold one call per block — anything
/// fancier should use a fenced fallback.
fn parse_one_body(body: &str) -> Option<ParsedToolCall> {
    parse_bodies(body).into_iter().next()
}

fn make_call(body: ToolCallBody) -> ParsedToolCall {
    ParsedToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        name: body.name,
        arguments: body.arguments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_tool_call() {
        let text = r#"Let me check that file.
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.rs"}}
</tool_call>"#;
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "read_file");
        assert_eq!(out.tool_calls[0].arguments["path"], "src/main.rs");
        assert_eq!(out.prose.trim(), "Let me check that file.");
    }

    #[test]
    fn extracts_multiple_calls() {
        let text = r#"<tool_call>{"name":"list_dir","arguments":{"path":"."}}</tool_call>
between
<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>
tail"#;
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].name, "list_dir");
        assert_eq!(out.tool_calls[1].name, "read_file");
        assert!(out.prose.contains("between"));
        assert!(out.prose.contains("tail"));
    }

    #[test]
    fn malformed_body_is_left_as_prose() {
        let text = r#"<tool_call>not json</tool_call>after"#;
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 0);
        assert!(out.prose.contains("not json"));
        assert!(out.prose.ends_with("after"));
    }

    #[test]
    fn unterminated_block_is_preserved() {
        let text = "preface <tool_call>still talking";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 0);
        assert!(out.prose.contains("preface"));
        assert!(out.prose.contains("still talking"));
    }

    #[test]
    fn pure_prose_passes_through() {
        let out = parse("hello world");
        assert_eq!(out.tool_calls.len(), 0);
        assert_eq!(out.prose, "hello world");
    }

    #[test]
    fn tolerates_codefence_wrappers() {
        let text = "<tool_call>```json\n{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"}}\n```</tool_call>";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "list_dir");
    }

    #[test]
    fn fenced_json_fallback_extracts_tool_call() {
        // Real-world output from Qwen2.5-Coder-7B with no <tool_call> tags.
        let text = "```json\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"intisharul.txt\", \"content\": \"intisharul\"}}\n```";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "write_file");
        assert_eq!(out.tool_calls[0].arguments["path"], "intisharul.txt");
        assert_eq!(out.prose, "");
    }

    #[test]
    fn fenced_json_with_surrounding_prose() {
        let text = "Sure, I'll do that.\n\n```json\n{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}\n```\n\nLet me know.";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "list_dir");
        // Prose is preserved around the extracted block.
        assert!(out.prose.contains("Sure"));
        assert!(out.prose.contains("Let me know"));
    }

    #[test]
    fn fenced_json_without_tool_shape_stays_prose() {
        // The model is showing the user an example JSON, not a tool call.
        let text = "Here's an example:\n```json\n{\"hello\": \"world\"}\n```";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 0);
        assert!(out.prose.contains("```json"));
    }

    #[test]
    fn xml_block_takes_precedence_over_fences() {
        // If both formats coexist, prefer the canonical one (don't extract twice).
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n```json\n{\"name\":\"b\",\"arguments\":{}}\n```";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].name, "a");
    }

    #[test]
    fn fenced_ndjson_extracts_multiple_calls() {
        // The exact payload Qwen2.5-Coder-7B produces when asked to scaffold
        // a Rust project: four objects, one per line, in one ```json fence.
        let text = "```json\n\
            {\"name\": \"create_dir\", \"arguments\": {\"path\": \"demo\"}}\n\
            {\"name\": \"create_dir\", \"arguments\": {\"path\": \"demo/src\"}}\n\
            {\"name\": \"write_file\", \"arguments\": {\"path\": \"demo/Cargo.toml\", \"content\": \"[package]\\nname = \\\"demo\\\"\\n\"}}\n\
            {\"name\": \"write_file\", \"arguments\": {\"path\": \"demo/src/main.rs\", \"content\": \"fn main() {}\"}}\n\
            ```";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 4);
        assert_eq!(out.tool_calls[0].name, "create_dir");
        assert_eq!(out.tool_calls[1].arguments["path"], "demo/src");
        assert_eq!(out.tool_calls[2].name, "write_file");
        assert_eq!(out.tool_calls[3].arguments["path"], "demo/src/main.rs");
        assert_eq!(out.prose, "");
    }

    #[test]
    fn fenced_json_array_extracts_multiple_calls() {
        // Some models wrap calls in a JSON array instead of NDJSON.
        let text = "```json\n[\
            {\"name\":\"a\",\"arguments\":{}},\
            {\"name\":\"b\",\"arguments\":{}}\
        ]\n```";
        let out = parse(text);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].name, "a");
        assert_eq!(out.tool_calls[1].name, "b");
    }
}
