# Agent Team Playground — Backend

FastAPI backend for multi-agent team orchestration. Agents communicate via filesystem-based JSON inboxes, coordinated by an orchestration loop that runs agent turns in parallel.

**Frontend repo:** [agent-team-playground-frontend](https://github.com/isscottw/agent-team-playground-frontend)

**Live frontend:** [agent-team-playground-frontend.vercel.app](https://agent-team-playground-frontend.vercel.app)

## Supported LLM Providers

- **Anthropic** (Claude) — recommended
- **OpenAI** (GPT)
- **Kimi / Moonshot** (K2)
- **Ollama** (local models)

## Getting Started

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

### Environment Variables

Create `.env`:

```
# At least one LLM provider key
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Supabase for session history persistence
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/sessions` | Create and start an agent team session |
| `DELETE` | `/api/sessions/{id}` | Stop and clean up a session |
| `POST` | `/api/sessions/{id}/chat` | Send a user message to the team |
| `GET` | `/api/sessions/{id}/stream` | SSE event stream for a session |
| `POST` | `/api/llm/test` | Validate an API key |
| `GET` | `/api/models` | Available models by provider |
| `GET` | `/api/history` | List past sessions (requires Supabase) |
| `GET` | `/api/history/{id}` | Full session replay data |
| `DELETE` | `/api/history/{id}` | Delete a session from history |
| `GET` | `/health` | Health check |

## Architecture

- **Orchestration loop** — Polls agent inboxes, runs agents with unread messages in parallel via `asyncio.gather`, leader nudge on idle, configurable timeout.
- **Agent turns** — Per-agent loop: check shutdown → compact history → build context → LLM call → process tool calls → repeat (up to 10 loops) → send idle notification.
- **Communication** — Filesystem JSON inboxes at `data/sessions/{id}/inboxes/{agent}.json`. Agents use `SendMessage`, `TaskCreate`, `TaskUpdate`, `TaskList`, `TaskGet` tools.
- **Supabase sync** — Fire-and-forget background sync of messages, tasks, and agent turns. No-op if env vars not set.

## Deployment

Deployed on **GCP Cloud Run**.

```bash
export CLOUDSDK_PYTHON=/usr/local/opt/python@3.13/libexec/bin/python3
gcloud run deploy agent-team-backend \
  --source . \
  --region us-east1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 1
```

Set environment variables via:

```bash
gcloud run services update agent-team-backend --region us-east1 \
  --set-env-vars "SUPABASE_URL=...,SUPABASE_KEY=..."
```
