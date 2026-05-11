-- Long-term memories: facts the assistant should remember across every
-- session. Injected into the system prompt at context-build time.

CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY NOT NULL,
    content    TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'manual', -- 'manual' or 'auto'
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
