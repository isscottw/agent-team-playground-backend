# Agent Team Playground - Backend

## Overview

FastAPI backend for multi-agent team orchestration with LLM providers. Agents communicate via filesystem-based JSON inboxes and a shared task store, coordinated by an orchestration loop that runs agent turns in parallel. Supports Anthropic, OpenAI, Kimi (Moonshot), and Ollama providers.

## Tech Stack

- **Framework**: FastAPI, Pydantic
- **LLM SDKs**: Anthropic, OpenAI, Kimi (Anthropic-compatible), Ollama (via `httpx`)
- **Persistence**: Filesystem JSON (primary), Supabase (optional, for history/replay)
- **Streaming**: Server-Sent Events (SSE) via `StreamingResponse`
- **Python**: 3.10+

## Dev Commands

```bash
# Start dev server
uvicorn main:app --reload --port 8080

# Run tests
pytest tests/

# Async tests require pytest-asyncio
pip install pytest-asyncio
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For Anthropic models | Claude API key |
| `OPENAI_API_KEY` | For OpenAI models | GPT API key |
| `KIMI_API_KEY` | For Kimi models | Moonshot AI API key |
| `SUPABASE_URL` | No | Supabase project URL (enables history persistence) |
| `SUPABASE_KEY` | No | Supabase service role key |

## Architecture

### Entry Point

- `main.py` -- FastAPI app, routes, session lifecycle, SSE streaming, global state (active sessions dict)

### Services (`services/`)

- `orchestration_service.py` -- `TeamEngine`: creates JSON stores, builds `AgentRunner` instances with hierarchy detection, runs the orchestration loop (polls inboxes, runs agents with unread messages in parallel via `asyncio.gather`), leader nudge on idle, idle timeout
- `agent_service.py` -- `AgentRunner`: per-agent turn loop (check shutdown -> compact history -> build context -> LLM call -> process tool calls -> repeat up to 10 loops -> send idle notification to lead)
- `context_service.py` -- `ContextBuilder`: assembles system prompt with scoped task list, team roster, connection descriptions, and role-specific instructions (leader / sub-leader / teammate). Injects unread inbox messages as the latest user message.
- `tool_service.py` -- `ToolExecutor`: implements 5 tools (`SendMessage`, `TaskCreate`, `TaskUpdate`, `TaskList`, `TaskGet`). `SendMessage` handles both standard messages (DM, broadcast) and protocol types (shutdown_request, shutdown_response, plan_approval_request, plan_approval_response).
- `protocol_service.py` -- `ProtocolService`: 7 JSON-in-JSON protocol message types: `shutdown_request`, `shutdown_approved`, `idle_notification`, `task_assignment`, `task_completed`, `plan_approval_request`, `plan_approval_response`. Includes `parse_protocol_message()` for detecting protocol messages in plain text fields.
- `model_service.py` -- `ModelService`: model registry with validation. Knows all models across all providers.
- `supabase_service.py` -- `SupabaseService`: best-effort write-only Supabase sync for session history. No-op if env vars not set.

### LLM Providers (`llm/`)

- `base.py` -- `LLMProvider` ABC, `LLMResponse` and `ToolCall` dataclasses
- `factory.py` -- `get_provider()` factory function
- `anthropic_provider.py`, `openai_provider.py`, `kimi_provider.py`, `ollama_provider.py` -- Provider implementations

### Communication Layer (`comms/`)

- `json_store.py` -- `InboxStore`: per-agent inbox JSON files at `data/sessions/{session_id}/inboxes/{agent}.json`. Async lock per agent, read/write/append/mark-read operations.
- `task_store.py` -- `TaskStore`: per-task JSON files at `data/sessions/{session_id}/tasks/{id}.json`. High-watermark ID counter, supports create/update/delete/list with blocking dependencies.
- `sync.py` -- `SupabaseSync`: fire-and-forget background sync of messages, tasks, and agent turns to Supabase.

### Other

- `models/` -- Pydantic request/response models (`SessionRequest`, `SessionResponse`, protocol models)
- `utils/sse.py` -- `SSEBroadcaster`: manages per-session SSE subscriber queues
- `utils/metrics.py` -- `TokenTracker`: per-session token usage tracking
- `supabase/migrations/` -- Database migration SQL files
- `demo.py` -- Standalone script for testing protocol flow without the full server

## Hierarchy Model

Role-based hierarchy determined by agent config:

- **Leader**: receives user messages, delegates to teammates. Has `role="leader"` and `lead_agent=None` (top leader) or `lead_agent=<parent>` (sub-leader).
- **Sub-leader**: a leader whose `connections` include a parent leader. Reports upward, delegates downward.
- **Teammate**: does assigned work, reports results back to their `lead_agent`.

Canvas arrows in the frontend define `connections`. The orchestration service uses connections to determine each agent's lead and scoped visibility.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions` | Create a new agent team session |
| `DELETE` | `/api/sessions/{id}` | Stop and clean up a session |
| `POST` | `/api/sessions/{id}/chat` | Send a user message to the team |
| `GET` | `/api/sessions/{id}/stream` | SSE event stream for a session |
| `POST` | `/api/llm/test` | Validate an API key with a minimal LLM call |
| `GET` | `/api/models` | Get available models grouped by provider |
| `GET` | `/api/history` | List past sessions (requires Supabase) |
| `GET` | `/api/history/{id}` | Get full replay data for a session |
| `GET` | `/health` | Health check |

## Tests

```bash
pytest tests/
```

Tests are in `tests/`. Async tests require `pytest-asyncio`. The main test file is `tests/test_protocol_integration.py`.

## Deployment

- **GCP Cloud Run**: Dockerfile is ready. Set `PORT` env var (defaults to 8080).
- **Supabase**: Optional. Set `SUPABASE_URL` and `SUPABASE_KEY` for session history persistence. Migrations in `supabase/migrations/`.
