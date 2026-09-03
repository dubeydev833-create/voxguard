"""Unit tests for the VoxGuard Agent Controller."""

import asyncio
import pytest

from app.models.state import SessionState
from app.models.tool import ToolResult
from app.services.agent_controller import AgentController
from app.services.session_manager import SessionManager
from app.tools.registry import create_default_registry


@pytest.fixture
def controller():
    mgr = SessionManager()
    reg = create_default_registry()
    ctrl = AgentController(manager=mgr, registry=reg)
    yield ctrl
    mgr.clear()


def test_intent_parsing_coverage(controller: AgentController):
    # Hotel search
    parsed = controller.default_intent_parser("Find a hotel in Paris with price under 2500")
    assert parsed is not None
    assert parsed[0] == "hotel_search"
    assert parsed[1]["max_price"] == 2500.0

    # Book ride
    parsed = controller.default_intent_parser("Book a ride from 1st Ave to SeaTac Airport")
    assert parsed is not None
    assert parsed[0] == "book_ride"
    assert "1st Ave" in parsed[1]["pickup_location"]
    assert "SeaTac Airport" in parsed[1]["dropoff_location"]

    # Weather
    parsed = controller.default_intent_parser("What is the weather in Chicago?")
    assert parsed is not None
    assert parsed[0] == "get_weather"
    assert parsed[1]["location"] == "Chicago"

    # Email
    parsed = controller.default_intent_parser("Send email to test@domain.com saying meeting at 5")
    assert parsed is not None
    assert parsed[0] == "send_email"
    assert parsed[1]["recipient"] == "test@domain.com"

    # Transfer funds
    parsed = controller.default_intent_parser("Please transfer $500 to acc_987654")
    assert parsed is not None
    assert parsed[0] == "transfer_funds"
    assert parsed[1]["amount"] == 500.0
    assert parsed[1]["recipient_account"] == "ACC_987654"

    # Device control
    parsed = controller.default_intent_parser("Turn on the living room lights")
    assert parsed is not None
    assert parsed[0] == "control_device"
    assert parsed[1]["action"] == "turn_on"

    # Direct conversation without tools
    parsed_chat = controller.default_intent_parser("Hello there, how are you?")
    assert parsed_chat is None


@pytest.mark.asyncio
async def test_handle_turn_with_tool_execution(controller: AgentController):
    session_id = "sess_ctrl_001"

    session = controller.handle_turn(
        session_id=session_id,
        transcript="What is the weather in Seattle?",
        simulated_delay=0.01,
    )

    assert session.current_version == 1
    assert session.state == SessionState.TOOL_RUNNING

    # Allow tool_worker task to complete
    if session.active_task:
        await session.active_task

    updated_session = controller.session_manager.get_session(session_id)
    assert updated_session.state == SessionState.COMPLETED
    assert updated_session.committed_version == 1
    assert "Seattle" in updated_session.last_response
    assert "weather" in updated_session.last_response.lower()


@pytest.mark.asyncio
async def test_handle_turn_conversational_no_tool(controller: AgentController):
    session_id = "sess_ctrl_002"

    session = controller.handle_turn(
        session_id=session_id,
        transcript="Hello, tell me what you can do.",
    )

    assert session.current_version == 1
    assert session.state == SessionState.COMPLETED
    assert "Hello, tell me what you can do." in session.last_response


def test_synthesize_response(controller: AgentController):
    tool_res = ToolResult.ok(
        tool_name="hotel_search",
        output={"destination": "Downtown", "max_price": 3000, "count": 2},
    )
    resp = controller.synthesize_response("hotel_search", tool_res)
    assert "2 hotels in Downtown under $3000" in resp

    fail_res = ToolResult.fail(tool_name="book_ride", error="No drivers nearby")
    fail_resp = controller.synthesize_response("book_ride", fail_res)
    assert "error executing book_ride" in fail_resp
