"""
Integration test for protocol gaps — exercises the full orchestration loop
with a mock LLM provider to verify:

1. Agent color assignment
2. idle_notification sent to lead after each turn
3. shutdown_request sent to all agents on stop()
4. Protocol messages parsed/labeled in agent context
5. task_assignment protocol message on TaskUpdate with owner
6. Lazy inbox creation (no directory until first write)
7. .lock file in task store
8. SSE events for protocol_message
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.base import LLMProvider, LLMResponse, ToolCall
from models.session_models import AgentConfig
from services.orchestration_service import TeamEngine, AGENT_COLORS
from services.protocol_service import ProtocolService
from comms.json_store import InboxStore
from comms.task_store import TaskStore


# ─── Mock LLM Provider ───────────────────────────────────────────────

class MockProvider(LLMProvider):
    """Returns scripted responses for testing."""

    def __init__(self):
        self._responses: List[LLMResponse] = []
        self._call_count = 0

    def queue(self, *responses: LLMResponse):
        self._responses.extend(responses)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: str = "",
        model: Optional[str] = None,
    ) -> LLMResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = LLMResponse(content="(no more scripted responses)")
        self._call_count += 1
        return resp


# ─── Helpers ──────────────────────────────────────────────────────────

DATA_DIR = Path("data/test_sessions")


def cleanup_test_data(session_id: str):
    import shutil
    d = DATA_DIR.parent / "sessions" / session_id
    if d.exists():
        shutil.rmtree(d)


# ─── Tests ────────────────────────────────────────────────────────────

async def test_agent_colors():
    """Verify agents get auto-assigned colors from the palette."""
    print("\n═══ Test 1: Agent Color Assignment ═══")

    agents = [
        AgentConfig(name="lead", provider="anthropic", model="test"),
        AgentConfig(name="worker-a", provider="anthropic", model="test"),
        AgentConfig(name="worker-b", provider="anthropic", model="test"),
    ]
    engine = TeamEngine(
        session_id="test-colors",
        agents=agents,
        api_keys={"anthropic": "fake"},
    )
    engine._build_runners()

    for i, name in enumerate(["lead", "worker-a", "worker-b"]):
        runner = engine.runners[name]
        expected_color = AGENT_COLORS[i % len(AGENT_COLORS)]
        assert runner.color == expected_color, f"{name}: expected {expected_color}, got {runner.color}"
        print(f"  ✓ {name} → color={runner.color}")

    # lead_agent should be the first agent for all runners
    for name, runner in engine.runners.items():
        assert runner.lead_agent == "lead", f"{name}: lead_agent should be 'lead', got {runner.lead_agent}"
    print("  ✓ All runners have lead_agent='lead'")

    cleanup_test_data("test-colors")
    print("  PASSED")


async def test_lazy_inbox():
    """Verify inbox directory is NOT created until first write."""
    print("\n═══ Test 2: Lazy Inbox Creation ═══")

    session_id = "test-lazy-inbox"
    store = InboxStore(session_id)
    inbox_dir = store.inbox_dir

    # Directory should NOT exist yet
    assert not inbox_dir.exists(), f"Inbox dir should not exist yet, but found: {inbox_dir}"
    print("  ✓ Inbox directory does NOT exist after __init__")

    # Reading should return empty without creating dir
    msgs = await store.read_all("agent-a")
    assert msgs == [], f"Expected empty list, got {msgs}"
    assert not inbox_dir.exists(), "read_all should not create directory"
    print("  ✓ read_all returns [] without creating directory")

    unread = await store.read_unread("agent-a")
    assert unread == [], f"Expected empty list, got {unread}"
    assert not inbox_dir.exists(), "read_unread should not create directory"
    print("  ✓ read_unread returns [] without creating directory")

    # Writing should create directory
    await store.append_message(agent="agent-a", from_agent="user", text="hello")
    assert inbox_dir.exists(), "append_message should create directory"
    print("  ✓ append_message creates inbox directory on first write")

    # Agent-b should NOT have an inbox file (lazy per-agent)
    agent_b_path = inbox_dir / "agent-b.json"
    assert not agent_b_path.exists(), "agent-b should not have inbox file yet"
    print("  ✓ Only agent-a has an inbox file (agent-b does not)")

    cleanup_test_data(session_id)
    print("  PASSED")


async def test_task_store_lock_file():
    """Verify .lock file is created in task directory."""
    print("\n═══ Test 3: Task Store .lock File ═══")

    session_id = "test-lock-file"
    store = TaskStore(session_id)

    lock_path = store.task_dir / ".lock"
    assert lock_path.exists(), ".lock file should exist"
    assert lock_path.read_text() == "", ".lock should be empty"
    print("  ✓ .lock file exists and is empty")

    hwm_path = store.task_dir / ".highwatermark"
    assert hwm_path.exists(), ".highwatermark should exist"
    print("  ✓ .highwatermark file exists")

    store.cleanup()
    print("  PASSED")


async def test_idle_notification_after_turn():
    """Verify idle_notification is sent to lead's inbox after agent turn."""
    print("\n═══ Test 4: Idle Notification After Turn ═══")

    session_id = "test-idle-notif"
    inbox = InboxStore(session_id)
    tasks = TaskStore(session_id)

    # Collect SSE events
    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    from services.agent_service import AgentRunner

    runner = AgentRunner(
        agent_name="worker",
        provider_name="anthropic",
        model="test",
        api_key="fake",
        system_prompt="You are a test worker.",
        inbox_store=inbox,
        task_store=tasks,
        team_agents=["lead", "worker"],
        emit_sse=mock_sse,
        lead_agent="lead",
        color="green",
    )

    # Inject a mock provider that returns a simple response
    mock = MockProvider()
    mock.queue(LLMResponse(content="I'm done with my work."))
    runner.provider = mock

    # Send initial message to trigger turn
    await inbox.append_message(agent="worker", from_agent="user", text="Do something")

    # Run turn
    await runner.run_turn()

    # Check lead's inbox for idle_notification
    lead_msgs = await inbox.read_all("lead")
    idle_msgs = []
    for m in lead_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "idle_notification":
            idle_msgs.append(parsed)

    assert len(idle_msgs) == 1, f"Expected 1 idle_notification in lead inbox, found {len(idle_msgs)}"
    assert idle_msgs[0]["from"] == "worker"
    print(f"  ✓ idle_notification found in lead inbox: from={idle_msgs[0]['from']}")

    # Check SSE events for protocol_message
    protocol_events = [e for e in sse_events if e.get("type") == "protocol_message"]
    assert len(protocol_events) >= 1, f"Expected protocol_message SSE event, got {len(protocol_events)}"
    pe = protocol_events[0]
    assert pe["data"]["protocol_type"] == "idle_notification"
    print(f"  ✓ SSE protocol_message emitted: {pe['data']}")

    cleanup_test_data(session_id)
    tasks.cleanup()
    print("  PASSED")


async def test_protocol_message_parsing_in_context():
    """Verify context builder labels protocol messages differently from plain messages."""
    print("\n═══ Test 5: Protocol Message Parsing in Context ═══")

    session_id = "test-proto-context"
    inbox = InboxStore(session_id)
    tasks = TaskStore(session_id)

    from services.context_service import ContextBuilder

    builder = ContextBuilder(
        inbox_store=inbox,
        task_store=tasks,
        agent_name="lead",
        agent_system_prompt="You are the team lead.",
        team_agents=["lead", "worker"],
    )

    # Send a plain message
    await inbox.append_message(agent="lead", from_agent="worker", text="Here's the research result.")

    # Send a protocol message (idle_notification)
    idle_msg = ProtocolService.create_idle_notification(from_agent="worker")
    await inbox.append_message(agent="lead", from_agent="worker", text=idle_msg["text"], summary=idle_msg["summary"])

    # Build messages
    messages = await builder.build_messages([])

    # The last message should contain inbox content
    inbox_content = messages[-1]["content"]
    print(f"  Inbox context block:\n    {inbox_content.replace(chr(10), chr(10) + '    ')}")

    assert "[Message from worker]: Here's the research result." in inbox_content
    assert "[Protocol: idle_notification from worker]" in inbox_content
    assert "idle_notification" not in inbox_content.split("[Protocol:")[0].split("[Message")[1]  # plain msg doesn't contain protocol
    print("  ✓ Plain message labeled as [Message from ...]")
    print("  ✓ Protocol message labeled as [Protocol: ... from ...]")

    cleanup_test_data(session_id)
    tasks.cleanup()
    print("  PASSED")


async def test_task_assignment_protocol():
    """Verify TaskUpdate with owner sends task_assignment protocol message."""
    print("\n═══ Test 6: Task Assignment Protocol Message ═══")

    session_id = "test-task-assign"
    inbox = InboxStore(session_id)
    task_store = TaskStore(session_id)

    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    from services.agent_service import AgentRunner

    runner = AgentRunner(
        agent_name="lead",
        provider_name="anthropic",
        model="test",
        api_key="fake",
        system_prompt="You are the lead.",
        inbox_store=inbox,
        task_store=task_store,
        team_agents=["lead", "worker"],
        emit_sse=mock_sse,
        lead_agent="lead",
        color="blue",
    )

    # Create a task first
    task = await task_store.create_task(subject="Write tests", description="Write integration tests")

    # Now use the tool executor to assign it
    result_str = await runner.tool_executor.execute("TaskUpdate", {
        "taskId": task["id"],
        "owner": "worker",
    })
    result = json.loads(result_str)
    assert result.get("owner") == "worker", f"Task owner should be 'worker', got {result.get('owner')}"
    print(f"  ✓ Task #{task['id']} assigned to worker")

    # Check worker's inbox for task_assignment protocol message
    worker_msgs = await inbox.read_all("worker")
    assignment_msgs = []
    for m in worker_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "task_assignment":
            assignment_msgs.append(parsed)

    assert len(assignment_msgs) == 1, f"Expected 1 task_assignment, found {len(assignment_msgs)}"
    assert assignment_msgs[0]["taskId"] == task["id"]
    assert assignment_msgs[0]["taskSubject"] == "Write tests"
    print(f"  ✓ task_assignment in worker inbox: taskId={assignment_msgs[0]['taskId']}, subject={assignment_msgs[0]['taskSubject']}")

    # Check SSE
    proto_events = [e for e in sse_events if e.get("type") == "protocol_message" and e.get("data", {}).get("protocol_type") == "task_assignment"]
    assert len(proto_events) >= 1
    print(f"  ✓ SSE protocol_message emitted for task_assignment")

    cleanup_test_data(session_id)
    task_store.cleanup()
    print("  PASSED")


async def test_shutdown_protocol():
    """Verify stop() sends shutdown_request to all agents."""
    print("\n═══ Test 7: Shutdown Protocol ═══")

    session_id = "test-shutdown"
    agents = [
        AgentConfig(name="lead", provider="anthropic", model="test"),
        AgentConfig(name="worker", provider="anthropic", model="test"),
    ]

    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    engine = TeamEngine(
        session_id=session_id,
        agents=agents,
        api_keys={"anthropic": "fake"},
        emit_sse=mock_sse,
    )
    engine._build_runners()
    engine._running = True

    # Call stop
    await engine.stop()

    # Check each agent's inbox for shutdown_request
    for name in ["lead", "worker"]:
        msgs = await engine.inbox_store.read_all(name)
        shutdown_msgs = []
        for m in msgs:
            parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
            if parsed and parsed.get("type") == "shutdown_request":
                shutdown_msgs.append(parsed)
        assert len(shutdown_msgs) == 1, f"{name}: expected 1 shutdown_request, found {len(shutdown_msgs)}"
        assert shutdown_msgs[0]["reason"] == "session ending"
        print(f"  ✓ {name} received shutdown_request (reason='session ending')")

    # Check SSE
    shutdown_sse = [e for e in sse_events if e.get("type") == "protocol_message" and e.get("data", {}).get("protocol_type") == "shutdown_request"]
    assert len(shutdown_sse) >= 1
    print(f"  ✓ SSE protocol_message emitted for shutdown_request")

    cleanup_test_data(session_id)
    print("  PASSED")


async def test_full_orchestration_round():
    """End-to-end: start session, send message, agent responds, idle notification sent, then stop."""
    print("\n═══ Test 8: Full Orchestration Round ═══")

    session_id = "test-full-round"

    agents = [
        AgentConfig(name="lead", provider="anthropic", model="test"),
        AgentConfig(name="observer", provider="anthropic", model="test"),
    ]

    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    engine = TeamEngine(
        session_id=session_id,
        agents=agents,
        api_keys={"anthropic": "fake"},
        emit_sse=mock_sse,
    )

    # Patch all runners with mock providers after build
    engine._build_runners()

    # Lead will: respond with text + use SendMessage to observer
    lead_mock = MockProvider()
    lead_mock.queue(
        # First call: LLM wants to send a message to observer
        LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc1", name="SendMessage", arguments={
                "type": "message",
                "recipient": "observer",
                "content": "Please observe the system.",
                "summary": "Asking observer to watch",
            })],
        ),
        # Second call: after tool result, produce final text
        LLMResponse(content="I've asked the observer to watch the system."),
    )
    engine.runners["lead"].provider = lead_mock

    # Observer will: just respond with text
    observer_mock = MockProvider()
    observer_mock.queue(
        LLMResponse(content="Acknowledged. Observing the system now."),
    )
    engine.runners["observer"].provider = observer_mock

    # Send user message to lead
    await engine.send_user_message("Start monitoring the system.")

    # Manually run one round of the orchestration loop (not the full loop — just process pending messages)
    # Check lead has unread
    lead_msgs = await engine.inbox_store.read_all("lead")
    assert any(not m.get("read") for m in lead_msgs), "Lead should have unread messages"
    print("  ✓ User message delivered to lead's inbox")

    # Run lead's turn
    await engine.runners["lead"].run_turn()
    print("  ✓ Lead's turn completed (sent message to observer via tool)")

    # Verify observer got the message
    observer_msgs = await engine.inbox_store.read_all("observer")
    plain_msgs = [m for m in observer_msgs if not ProtocolService.parse_protocol_message(m.get("text", ""))]
    assert len(plain_msgs) >= 1, "Observer should have received a plain message from lead"
    print(f"  ✓ Observer received message: '{plain_msgs[0]['text'][:50]}...'")

    # Verify idle_notification was sent to lead (since observer is not lead)
    # But actually lead IS the lead, so after lead's turn, idle_notification is NOT sent (lead != lead guard)
    # Let's run observer's turn instead
    await engine.runners["observer"].run_turn()
    print("  ✓ Observer's turn completed")

    # Now check lead's inbox for idle_notification from observer
    lead_msgs_after = await engine.inbox_store.read_all("lead")
    idle_from_observer = []
    for m in lead_msgs_after:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "idle_notification" and parsed.get("from") == "observer":
            idle_from_observer.append(parsed)

    assert len(idle_from_observer) >= 1, "Lead should have idle_notification from observer"
    print(f"  ✓ Lead received idle_notification from observer")

    # Check SSE events
    event_types = [e.get("type") for e in sse_events]
    assert "turn_start" in event_types
    assert "agent_response" in event_types
    assert "tool_call" in event_types
    assert "agent_message" in event_types
    assert "protocol_message" in event_types
    print(f"  ✓ SSE events emitted: {set(event_types)}")

    # Now stop and verify shutdown
    await engine.stop()
    for name in ["lead", "observer"]:
        msgs = await engine.inbox_store.read_all(name)
        has_shutdown = any(
            ProtocolService.parse_protocol_message(m.get("text", "")) and
            ProtocolService.parse_protocol_message(m.get("text", "")).get("type") == "shutdown_request"
            for m in msgs
        )
        assert has_shutdown, f"{name} should have shutdown_request"
    print("  ✓ Both agents received shutdown_request on stop()")

    # Verify filesystem layout
    inbox_dir = engine.inbox_store.inbox_dir
    assert inbox_dir.exists(), "Inbox directory should exist (messages were written)"
    task_dir = engine.task_store.task_dir
    assert (task_dir / ".lock").exists(), ".lock file should exist in task dir"
    assert (task_dir / ".highwatermark").exists(), ".highwatermark should exist"
    print(f"  ✓ Filesystem layout: {inbox_dir} exists with inbox files")
    print(f"  ✓ Filesystem layout: {task_dir}/.lock exists")

    cleanup_test_data(session_id)
    engine.task_store.cleanup()
    print("  PASSED")


# ─── Round 2 Tests ────────────────────────────────────────────────────

async def test_metadata_and_deleted_status():
    """Verify metadata field on create/update and deleted status removes task file."""
    print("\n═══ Test 9: Metadata + Deleted Status ═══")

    session_id = "test-metadata"
    store = TaskStore(session_id)

    # Create with metadata
    task = await store.create_task(
        subject="Research API",
        description="Research the API docs",
        metadata={"priority": "high", "source": "user"},
    )
    assert task["metadata"] == {"priority": "high", "source": "user"}
    print(f"  ✓ Task created with metadata: {task['metadata']}")

    # Update metadata — merge + delete key
    updated = await store.update_task(task["id"], {
        "metadata": {"priority": "low", "source": None, "tag": "v2"},
    })
    assert updated["metadata"]["priority"] == "low"
    assert "source" not in updated["metadata"]
    assert updated["metadata"]["tag"] == "v2"
    print(f"  ✓ Metadata merged/deleted: {updated['metadata']}")

    # Delete via status
    task_path = store._task_path(task["id"])
    assert task_path.exists(), "Task file should exist before delete"
    result = await store.update_task(task["id"], {"status": "deleted"})
    assert result["deleted"] is True
    assert not task_path.exists(), "Task file should be removed after deleted status"
    print(f"  ✓ Task file removed by status='deleted'")

    store.cleanup()
    print("  PASSED")


async def test_shutdown_approved_protocol():
    """Verify shutdown auto-approve: agent receives shutdown_request, auto-sends shutdown_approved."""
    print("\n═══ Test 10: Shutdown Approved Auto-Response ═══")

    session_id = "test-shutdown-approved"
    inbox = InboxStore(session_id)
    tasks = TaskStore(session_id)

    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    from services.agent_service import AgentRunner

    runner = AgentRunner(
        agent_name="worker",
        provider_name="anthropic",
        model="test",
        api_key="fake",
        system_prompt="You are a worker.",
        inbox_store=inbox,
        task_store=tasks,
        team_agents=["lead", "worker"],
        emit_sse=mock_sse,
        lead_agent="lead",
        color="green",
    )

    # Inject a mock provider (shouldn't be called since shutdown short-circuits)
    mock = MockProvider()
    runner.provider = mock

    # Send shutdown_request to worker's inbox
    shutdown_msg = ProtocolService.create_shutdown_request(from_agent="lead", reason="session ending")
    await inbox.append_message(agent="worker", from_agent="lead", text=shutdown_msg["text"], summary=shutdown_msg["summary"])

    # Run turn — should auto-approve and return early
    result = await runner.run_turn()
    assert result.get("shutdown") is True, f"Expected shutdown=True, got {result}"
    assert mock._call_count == 0, "LLM should NOT be called when shutdown is auto-approved"
    print(f"  ✓ run_turn() returned early with shutdown=True, LLM not called")

    # Check lead's inbox for shutdown_approved
    lead_msgs = await inbox.read_all("lead")
    approved_msgs = []
    for m in lead_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "shutdown_approved":
            approved_msgs.append(parsed)

    assert len(approved_msgs) == 1, f"Expected 1 shutdown_approved in lead inbox, found {len(approved_msgs)}"
    assert approved_msgs[0]["from"] == "worker"
    print(f"  ✓ shutdown_approved found in lead inbox: from={approved_msgs[0]['from']}")

    # Check SSE
    proto_events = [e for e in sse_events if e.get("data", {}).get("protocol_type") == "shutdown_approved"]
    assert len(proto_events) >= 1
    print(f"  ✓ SSE protocol_message emitted for shutdown_approved")

    cleanup_test_data(session_id)
    tasks.cleanup()
    print("  PASSED")


async def test_task_completed_protocol():
    """Verify TaskUpdate to completed sends task_completed protocol to lead."""
    print("\n═══ Test 11: Task Completed Protocol ═══")

    session_id = "test-task-completed"
    inbox = InboxStore(session_id)
    task_store = TaskStore(session_id)

    sse_events: List[Dict] = []
    async def mock_sse(event):
        sse_events.append(event)

    from services.agent_service import AgentRunner

    runner = AgentRunner(
        agent_name="worker",
        provider_name="anthropic",
        model="test",
        api_key="fake",
        system_prompt="You are a worker.",
        inbox_store=inbox,
        task_store=task_store,
        team_agents=["lead", "worker"],
        emit_sse=mock_sse,
        lead_agent="lead",
        color="green",
    )

    # Create a task and assign to worker
    task = await task_store.create_task(subject="Write docs", description="Write the docs", owner="worker")

    # Use tool executor to mark completed
    result_str = await runner.tool_executor.execute("TaskUpdate", {
        "taskId": task["id"],
        "status": "completed",
    })
    result = json.loads(result_str)
    assert result.get("status") == "completed"
    print(f"  ✓ Task #{task['id']} marked completed")

    # Check lead's inbox for task_completed
    lead_msgs = await inbox.read_all("lead")
    completed_msgs = []
    for m in lead_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "task_completed":
            completed_msgs.append(parsed)

    assert len(completed_msgs) == 1, f"Expected 1 task_completed, found {len(completed_msgs)}"
    assert completed_msgs[0]["taskId"] == task["id"]
    assert completed_msgs[0]["taskSubject"] == "Write docs"
    print(f"  ✓ task_completed in lead inbox: taskId={completed_msgs[0]['taskId']}")

    # Check SSE
    proto_events = [e for e in sse_events if e.get("data", {}).get("protocol_type") == "task_completed"]
    assert len(proto_events) >= 1
    print(f"  ✓ SSE protocol_message emitted for task_completed")

    cleanup_test_data(session_id)
    task_store.cleanup()
    print("  PASSED")


async def test_plan_approval_flow():
    """Verify plan_approval_request and plan_approval_response via SendMessage."""
    print("\n═══ Test 12: Plan Approval Flow ═══")

    session_id = "test-plan-approval"
    inbox = InboxStore(session_id)
    task_store = TaskStore(session_id)

    from services.tool_service import ToolExecutor

    executor = ToolExecutor(
        inbox_store=inbox,
        task_store=task_store,
        agent_name="worker",
        team_agents=["lead", "worker"],
    )

    # Worker sends plan_approval_request to lead
    result_str = await executor.execute("SendMessage", {
        "type": "plan_approval_request",
        "recipient": "lead",
        "request_id": "req-001",
        "content": "I plan to refactor the auth module into three separate files.",
    })
    result = json.loads(result_str)
    assert result["status"] == "plan_approval_request_sent"
    assert result["request_id"] == "req-001"
    print(f"  ✓ plan_approval_request sent to lead")

    # Check lead's inbox
    lead_msgs = await inbox.read_all("lead")
    plan_reqs = []
    for m in lead_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "plan_approval_request":
            plan_reqs.append(parsed)

    assert len(plan_reqs) == 1
    assert plan_reqs[0]["requestId"] == "req-001"
    assert "refactor" in plan_reqs[0]["plan"]
    print(f"  ✓ plan_approval_request in lead inbox with correct requestId and plan text")

    # Lead approves
    lead_executor = ToolExecutor(
        inbox_store=inbox,
        task_store=task_store,
        agent_name="lead",
        team_agents=["lead", "worker"],
    )
    result_str = await lead_executor.execute("SendMessage", {
        "type": "plan_approval_response",
        "recipient": "worker",
        "request_id": "req-001",
        "approve": True,
        "content": "Looks good, proceed.",
    })
    result = json.loads(result_str)
    assert result["status"] == "plan_approval_response_sent"
    assert result["approve"] is True
    print(f"  ✓ plan_approval_response sent (approved)")

    # Check worker's inbox for approval
    worker_msgs = await inbox.read_all("worker")
    approvals = []
    for m in worker_msgs:
        parsed = ProtocolService.parse_protocol_message(m.get("text", ""))
        if parsed and parsed.get("type") == "plan_approval_response":
            approvals.append(parsed)

    assert len(approvals) == 1
    assert approvals[0]["approve"] is True
    assert approvals[0]["requestId"] == "req-001"
    print(f"  ✓ plan_approval_response in worker inbox: approved=True")

    cleanup_test_data(session_id)
    task_store.cleanup()
    print("  PASSED")


async def test_compaction_recovery():
    """Verify context compaction trims history and adds summary marker."""
    print("\n═══ Test 13: Compaction Recovery ═══")

    session_id = "test-compaction"
    inbox = InboxStore(session_id)
    tasks = TaskStore(session_id)

    from services.agent_service import AgentRunner, MAX_HISTORY_MESSAGES

    runner = AgentRunner(
        agent_name="worker",
        provider_name="anthropic",
        model="test",
        api_key="fake",
        system_prompt="You are a worker.",
        inbox_store=inbox,
        task_store=tasks,
        team_agents=["lead", "worker"],
        lead_agent="lead",
    )

    # Fill history beyond MAX_HISTORY_MESSAGES
    for i in range(50):
        role = "assistant" if i % 2 == 0 else "user"
        runner.conversation_history.append({"role": role, "content": f"Message {i}"})

    assert len(runner.conversation_history) == 50
    print(f"  ✓ History filled to {len(runner.conversation_history)} messages")

    # Trigger compaction
    runner._maybe_compact_history()

    # Should have summary marker + last 20
    assert len(runner.conversation_history) == 21, f"Expected 21 messages after compaction, got {len(runner.conversation_history)}"
    assert "compacted" in runner.conversation_history[0]["content"]
    assert runner.conversation_history[-1]["content"] == "Message 49"
    print(f"  ✓ History compacted to {len(runner.conversation_history)} messages (1 marker + 20 recent)")
    print(f"  ✓ Marker: '{runner.conversation_history[0]['content'][:60]}...'")

    # No compaction when under limit
    runner._maybe_compact_history()
    assert len(runner.conversation_history) == 21, "Should not compact again when under limit"
    print(f"  ✓ No re-compaction when under limit")

    cleanup_test_data(session_id)
    tasks.cleanup()
    print("  PASSED")


async def test_protocol_message_schemas():
    """Verify all 7 protocol message types produce valid JSON-in-JSON payloads."""
    print("\n═══ Test 14: All Protocol Message Schemas ═══")

    # Test all protocol message creators
    msgs = {
        "idle_notification": ProtocolService.create_idle_notification("worker"),
        "shutdown_request": ProtocolService.create_shutdown_request("lead", "ending"),
        "shutdown_approved": ProtocolService.create_shutdown_approved("worker"),
        "task_assignment": ProtocolService.create_task_assignment("lead", "1", "Write tests"),
        "task_completed": ProtocolService.create_task_completed("worker", "1", "Write tests"),
        "plan_approval_request": ProtocolService.create_plan_approval_request("worker", "req-1", "my plan"),
        "plan_approval_response": ProtocolService.create_plan_approval_response("lead", "req-1", True, "ok"),
    }

    for ptype, msg in msgs.items():
        # Verify envelope structure
        assert "from" in msg, f"{ptype}: missing 'from' field"
        assert "text" in msg, f"{ptype}: missing 'text' field"
        assert "timestamp" in msg, f"{ptype}: missing 'timestamp' field"

        # Verify JSON-in-JSON
        parsed = ProtocolService.parse_protocol_message(msg["text"])
        assert parsed is not None, f"{ptype}: text is not valid JSON"
        assert parsed["type"] == ptype, f"{ptype}: parsed type mismatch: {parsed['type']}"
        assert parsed["from"] in ("worker", "lead"), f"{ptype}: unexpected from: {parsed['from']}"
        print(f"  ✓ {ptype}: valid envelope + JSON-in-JSON payload")

    print(f"  ✓ All 7/7 protocol message types validated")
    print("  PASSED")


# ─── Runner ───────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("Protocol Integration Tests")
    print("=" * 60)

    tests = [
        test_agent_colors,
        test_lazy_inbox,
        test_task_store_lock_file,
        test_idle_notification_after_turn,
        test_protocol_message_parsing_in_context,
        test_task_assignment_protocol,
        test_shutdown_protocol,
        test_full_orchestration_round,
        # Round 2 tests
        test_metadata_and_deleted_status,
        test_shutdown_approved_protocol,
        test_task_completed_protocol,
        test_plan_approval_flow,
        test_compaction_recovery,
        test_protocol_message_schemas,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
