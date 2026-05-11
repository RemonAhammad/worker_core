//! Tool catalog and protocol used by the agent endpoint.
//!
//! The backend does NOT execute tools — it only declares their existence
//! to the model and parses `<tool_call>` blocks the model emits. Actual
//! execution happens on the client (the Tauri desktop app), which is the
//! trust anchor with the filesystem.
//!
//! The format follows the Qwen2.5-Coder Hermes-style tool calling:
//!
//! ```text
//! <tool_call>
//! {"name": "<fn>", "arguments": {...}}
//! </tool_call>
//! ```

pub mod parser;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One tool entry as the model sees it inside the `<tools>` system block.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

/// The full filesystem tool surface offered to the agent.
///
/// Keeping these as `&'static` since they're compile-time constants and
/// the JSON schema for each is straightforward to handcraft.
pub fn filesystem_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "list_dir",
                description: "List entries in a directory inside the workspace. Returns names and whether each is a file or directory.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace-relative path. Use '.' for the workspace root." }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "read_file",
                description: "Read a UTF-8 text file from the workspace. Caps at 256 KiB by default; bump max_bytes for larger files.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Workspace-relative file path." },
                        "max_bytes": { "type": "integer", "description": "Optional byte cap. Default 262144." }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "write_file",
                description: "Overwrite or create a UTF-8 text file inside the workspace. The user MUST approve every call.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "append_file",
                description: "Append text to an existing file. Creates the file if it does not exist. Requires user approval.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "delete_path",
                description: "Delete a file, or an empty directory. Refuses on non-empty directories. Requires user approval.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "move_path",
                description: "Rename or move a path within the workspace. Requires user approval.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "string" },
                        "to":   { "type": "string" }
                    },
                    "required": ["from", "to"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "create_dir",
                description: "Create a directory (and any missing parents) inside the workspace. Requires user approval.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    },
                    "required": ["path"]
                }),
            },
        },
    ]
}

/// Render the system prompt addendum that teaches the model about tools.
/// Caller concatenates this onto the existing session system prompt.
pub fn render_tool_preamble(tools: &[ToolDefinition], workspace_hint: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str("\n\n# Tools\n\nYou are running as a coding agent with access to the user's workspace via the following functions. Invoke at most one tool per <tool_call> block; the runtime will execute it and return the result to you.\n");
    if let Some(ws) = workspace_hint {
        s.push_str(&format!(
            "\nWorkspace root (read-only metadata, all paths are workspace-relative): `{}`\n",
            ws
        ));
    }
    s.push_str(
        "\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for t in tools {
        if let Ok(line) = serde_json::to_string(t) {
            s.push_str(&line);
            s.push('\n');
        }
    }
    s.push_str("</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\nWhen you are done — no further tool calls needed — reply to the user in plain prose.\n");
    s
}

/// Structured tool call extracted from model output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolCall {
    /// Synthetic id we generate so the client can match a result back to a
    /// call (the model doesn't produce ids itself).
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// What the client returns after running a tool. `ok=false` means the call
/// failed (or the user denied it); `content` should be a short textual
/// explanation in either case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub ok: bool,
    #[serde(default)]
    pub content: String,
}

/// Render a tool result the model can consume as the next assistant turn's
/// `<tool_response>` block.
pub fn render_tool_response(result: &ToolResult) -> String {
    let status = if result.ok { "ok" } else { "error" };
    format!(
        "<tool_response>\n{{\"id\": \"{}\", \"status\": \"{}\", \"content\": {}}}\n</tool_response>",
        result.id,
        status,
        serde_json::to_string(&result.content).unwrap_or_else(|_| "\"\"".to_string()),
    )
}
