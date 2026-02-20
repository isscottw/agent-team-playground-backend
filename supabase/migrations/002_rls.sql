-- Enable Row Level Security
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks_history ENABLE ROW LEVEL SECURITY;

-- Allow the service role full access (backend uses service key)
CREATE POLICY "Service role full access on sessions"
    ON sessions FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access on messages_history"
    ON messages_history FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access on agent_turns"
    ON agent_turns FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access on tasks_history"
    ON tasks_history FOR ALL
    USING (true)
    WITH CHECK (true);

-- Anonymous read access for history endpoints
CREATE POLICY "Anon read sessions"
    ON sessions FOR SELECT
    USING (true);

CREATE POLICY "Anon read messages"
    ON messages_history FOR SELECT
    USING (true);

CREATE POLICY "Anon read turns"
    ON agent_turns FOR SELECT
    USING (true);

CREATE POLICY "Anon read tasks"
    ON tasks_history FOR SELECT
    USING (true);
