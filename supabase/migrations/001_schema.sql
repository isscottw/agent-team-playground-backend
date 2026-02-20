-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agents JSONB NOT NULL DEFAULT '[]',
    config JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

-- Message history (every message exchanged in a session)
CREATE TABLE IF NOT EXISTS messages_history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    from_agent TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    summary TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_messages_session ON messages_history(session_id);
CREATE INDEX idx_messages_timestamp ON messages_history(timestamp);

-- Agent turns (LLM calls, tool calls, responses)
CREATE TABLE IF NOT EXISTS agent_turns (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    event_type TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_turns_session ON agent_turns(session_id);
CREATE INDEX idx_turns_timestamp ON agent_turns(timestamp);

-- Task snapshots (for replay)
CREATE TABLE IF NOT EXISTS tasks_history (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    owner TEXT,
    snapshot JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tasks_session ON tasks_history(session_id);
