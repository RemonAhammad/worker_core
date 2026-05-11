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
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "search",
                description: "Search file contents under the workspace for a substring or simple pattern. Returns up to `max_results` matching lines with their paths and line numbers. Case-sensitive by default.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Plain-text substring to search for. Not a regex."
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional workspace-relative subdirectory to scope the search. Defaults to the workspace root."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Cap on returned matches. Default 100; hard ceiling 1000."
                        },
                        "case_insensitive": {
                            "type": "boolean",
                            "description": "If true, fold case before comparing. Default false."
                        }
                    },
                    "required": ["query"]
                }),
            },
        },
        ToolDefinition {
            kind: "function",
            function: ToolFunction {
                name: "run_command",
                description: "Execute a single program inside the workspace and return its stdout, stderr, and exit code. PREFER this tool over write_file/create_dir for scaffolding (`flutter create`, `cargo new`, `npm init`, `git init`, …) and for build/test (`cargo test`, `npm run build`, `flutter pub get`) — never hand-write project files when a real command exists. Args are passed verbatim — there is NO shell, so no piping, redirects, &&, or globbing. Requires user approval. 30-second timeout by default.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Binary name or absolute path. Examples: `flutter`, `cargo`, `npm`, `git`, `python3`. NOT a shell snippet."
                        },
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Arguments passed to the binary. Each element becomes one argv entry. Example for `flutter create demo`: command=`flutter`, args=[`create`, `demo`]."
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Optional cap on runtime, in seconds. Default 30; hard ceiling 300."
                        },
                        "description": {
                            "type": "string",
                            "description": "Short human-readable explanation shown on the approval card so the user understands WHY this command runs. One sentence."
                        }
                    },
                    "required": ["command", "args"]
                }),
            },
        },
    ]
}

/// Render the system prompt addendum that teaches the model about tools.
///
/// Lives at the **top** of the final system block (above memories and the
/// session's own system prompt), and is written aggressively to keep weaker
/// models (e.g. Qwen2.5-Coder-7B) from sliding back into "let me suggest a
/// shell command" behavior. Concrete examples here matter — the model
/// pattern-matches them.
pub fn render_tool_preamble(tools: &[ToolDefinition], workspace_hint: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str(
        "# Filesystem agent mode\n\n\
You are operating as an autonomous coding agent. The user has granted you \
direct, runtime access to a workspace on their machine via the tools listed below. \
The runtime executes each tool you call and returns its result to you on the \
next turn.\n\n\
## CRITICAL RULES — follow them exactly\n\n\
1. When the user asks you to create, modify, read, move, search, or delete \
files, you MUST use the provided tools. Emit `<tool_call>` blocks; do NOT \
output shell commands as a response.\n\
2. NEVER suggest shell commands like `cargo new ...`, `echo ... > file`, \
`mkdir`, `rm`, or `mv` for the user to run. Those are not how you operate. \
Use the tools directly.\n\
3. NEVER ask 'would you like to proceed?' or 'shall I do this?' before \
calling a tool. Just call it. The runtime asks the human for explicit \
permission on mutating calls (`write_file`, `append_file`, `delete_path`, \
`move_path`, `create_dir`); you do not need to.\n\
4. For scaffolding, building, testing, version control, and any other task \
that has a real CLI command, USE `run_command` — do NOT try to hand-write \
project files. Examples (issue one tool call per turn):\n\
   - Rust crate: `run_command` with `{\"command\": \"cargo\", \"args\": [\"new\", \"demo\"], \"description\": \"scaffold a Rust crate named demo\"}`. Do NOT manually write Cargo.toml + src/main.rs; `cargo new` does it correctly.\n\
   - Flutter app: `run_command` with `{\"command\": \"flutter\", \"args\": [\"create\", \"demo\"], \"description\": \"scaffold a Flutter app named demo\"}`.\n\
   - Node project: `run_command` with `{\"command\": \"npm\", \"args\": [\"init\", \"-y\"]}`.\n\
   - Git: `run_command` with `{\"command\": \"git\", \"args\": [\"init\"]}`.\n\
   - Build/test: `run_command` with `{\"command\": \"cargo\", \"args\": [\"build\"]}` or `{\"command\": \"npm\", \"args\": [\"run\", \"build\"]}`.\n\
   Reach for `write_file` / `create_dir` only for files the user explicitly asks you to create or edit, or when no CLI exists.\n\
5. `run_command` is NOT a shell. You cannot use `|`, `&&`, `;`, `>`, or \
globs. Pass the binary as `command` and EACH argument as a separate \
element of `args`. To run two things, issue two separate tool calls in \
successive turns.\n\
6. All path arguments to FS tools are RELATIVE to the workspace root. Use \
`.` for the root itself. Do not include the absolute path the runtime \
mentions below.\n\
7. After the runtime returns a tool result, decide whether more calls are \
needed. When you have nothing left to do, reply to the user in plain prose \
summarizing what you did.\n\n",
    );
    if let Some(ws) = workspace_hint {
        s.push_str(&format!(
            "Workspace root (informational only; never include this prefix in tool arguments): `{}`\n\n",
            ws
        ));
    }
    s.push_str("Function signatures (JSON Schema) within <tools></tools>:\n<tools>\n");
    for t in tools {
        if let Ok(line) = serde_json::to_string(t) {
            s.push_str(&line);
            s.push('\n');
        }
    }
    s.push_str(
        "</tools>\n\nReturn each call as JSON within <tool_call></tool_call>:\n\
<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n",
    );
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
