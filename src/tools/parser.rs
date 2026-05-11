//! Extract `<tool_call>...</tool_call>` blocks from model output.
//!
//! The model is instructed to emit one JSON object per block in the form
//! `{"name": "...", "arguments": {...}}`. We tolerate whitespace and
//! line breaks around the JSON; we ignore (with a warning) any block whose
//! JSON we can't parse — better to surface the rest than reject the lot.
//!
//! The function also returns the "prose tail" of the output — everything
//! after the last `</tool_call>` — so the assistant message saved to the
//! DB only contains text the user should see.

use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use super::ParsedToolCall;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

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

    while let Some(rel) = output[cursor..].find(OPEN) {
        let open_idx = cursor + rel;
        // Prose preceding this tool call is preserved verbatim.
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
    ParseOutcome {
        prose: prose.trim().to_string(),
        tool_calls,
    }
}

fn parse_one_body(body: &str) -> Option<ParsedToolCall> {
    // Trim Markdown code-fence wrappers some models add: ```json {...} ```
    let stripped = body
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let parsed: ToolCallBody = serde_json::from_str(stripped).ok()?;
    Some(ParsedToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        name: parsed.name,
        arguments: parsed.arguments,
    })
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
}
