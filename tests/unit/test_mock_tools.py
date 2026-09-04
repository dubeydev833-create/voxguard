"""Unit tests for VoxGuard mock tools and tool registry."""

import pytest
from app.models.tool import RiskLevel, ToolDefinition, ToolResult
from app.tools.base import BaseTool
from app.tools.mock_tools import (
    MockBookRideTool,
    MockCalendarTool,
    MockDeviceControlTool,
    MockFlightSearchTool,
    MockGetWeatherTool,
    MockHotelSearchTool,
    MockRestaurantSearchTool,
    MockSendEmailTool,
    MockTransferFundsTool,
    get_mock_tools,
)
from app.tools.registry import ToolRegistry, create_default_registry, tool_registry


# ---------------------------------------------------------
# Test MockSendEmailTool
# ---------------------------------------------------------
def test_send_email_success():
    tool = MockSendEmailTool()
    assert tool.name == "send_email"
    assert tool.risk_level == RiskLevel.MEDIUM
    assert tool.requires_confirmation is True

    result = tool.run(
        recipient="user@example.com",
        subject="Meeting Follow-up",
        body="Here are the notes from our sync.",
        cc=["colleague@example.com"],
    )

    assert result.success is True
    assert result.tool_name == "send_email"
    assert result.output["status"] == "sent"
    assert result.output["recipient"] == "user@example.com"
    assert result.output["subject"] == "Meeting Follow-up"
    assert "notes from our sync" in result.output["body_preview"]
    assert result.output["message_id"].startswith("msg_")


def test_send_email_invalid_recipient():
    tool = MockSendEmailTool()
    result = tool.run(
        recipient="invalid-email-no-at",
        subject="Hello",
        body="Testing...",
    )
    assert result.success is False
    assert "Invalid email address" in result.error


def test_send_email_missing_required():
    tool = MockSendEmailTool()
    result = tool.run(recipient="user@example.com")
    assert result.success is False
    assert "Missing required parameter" in result.error


# ---------------------------------------------------------
# Test MockBookRideTool
# ---------------------------------------------------------
def test_book_ride_success():
    tool = MockBookRideTool()
    assert tool.name == "book_ride"
    assert tool.risk_level == RiskLevel.HIGH
    assert tool.requires_confirmation is True

    result = tool.run(
        pickup_location="100 Main St, San Francisco",
        dropoff_location="SFO Airport Terminal 2",
        ride_type="premium",
    )

    assert result.success is True
    assert result.output["status"] == "confirmed"
    assert result.output["ride_type"] == "premium"
    assert result.output["estimated_fare"] == 32.00
    assert result.output["pickup_location"] == "100 Main St, San Francisco"
    assert "driver" in result.output
    assert result.output["ride_id"].startswith("ride_")


def test_book_ride_missing_dropoff():
    tool = MockBookRideTool()
    result = tool.run(pickup_location="100 Main St")
    assert result.success is False
    assert "dropoff_location" in result.error


# ---------------------------------------------------------
# Test MockGetWeatherTool
# ---------------------------------------------------------
def test_get_weather_success():
    tool = MockGetWeatherTool()
    assert tool.name == "get_weather"
    assert tool.risk_level == RiskLevel.LOW
    assert tool.requires_confirmation is False

    result = tool.run(location="Seattle", units="fahrenheit")
    assert result.success is True
    assert result.output["location"] == "Seattle"
    assert result.output["units"] == "fahrenheit"
    assert result.output["temperature"] == 70
    assert "condition" in result.output


def test_get_weather_missing_location():
    tool = MockGetWeatherTool()
    result = tool.run()
    assert result.success is False
    assert "location" in result.error


# ---------------------------------------------------------
# Test MockTransferFundsTool
# ---------------------------------------------------------
def test_transfer_funds_success():
    tool = MockTransferFundsTool()
    assert tool.name == "transfer_funds"
    assert tool.risk_level == RiskLevel.CRITICAL
    assert tool.requires_confirmation is True

    result = tool.run(
        recipient_account="ACC_987654",
        amount=150.75,
        currency="USD",
    )
    assert result.success is True
    assert result.output["status"] == "completed"
    assert result.output["amount"] == 150.75
    assert result.output["recipient_account"] == "ACC_987654"
    assert result.output["transaction_id"].startswith("tx_")


def test_transfer_funds_invalid_amount():
    tool = MockTransferFundsTool()
    result = tool.run(recipient_account="ACC_987654", amount=-50)
    assert result.success is False
    assert "strictly positive" in result.error

    result_zero = tool.run(recipient_account="ACC_987654", amount=0)
    assert result_zero.success is False
    assert "strictly positive" in result_zero.error

    result_bad = tool.run(recipient_account="ACC_987654", amount="not-a-number")
    assert result_bad.success is False
    assert "must be a numeric value" in result_bad.error


# ---------------------------------------------------------
# Test MockCalendarTool
# ---------------------------------------------------------
def test_create_calendar_event_success():
    tool = MockCalendarTool()
    assert tool.name == "create_calendar_event"

    result = tool.run(
        title="VoxGuard Strategy Review",
        start_time="2026-09-10T10:00:00Z",
        end_time="2026-09-10T11:00:00Z",
        attendees=["alice@example.com", "bob@example.com"],
    )
    assert result.success is True
    assert result.output["status"] == "scheduled"
    assert result.output["title"] == "VoxGuard Strategy Review"
    assert len(result.output["attendees"]) == 2
    assert result.output["event_id"].startswith("evt_")


# ---------------------------------------------------------
# Test MockDeviceControlTool
# ---------------------------------------------------------
def test_control_device_success():
    tool = MockDeviceControlTool()
    assert tool.name == "control_device"
    assert tool.risk_level == RiskLevel.HIGH

    result = tool.run(device_id="front_door_lock", action="lock")
    assert result.success is True
    assert result.output["status"] == "success"
    assert result.output["device_id"] == "front_door_lock"
    assert result.output["action"] == "lock"


def test_control_device_unsupported_action():
    tool = MockDeviceControlTool()
    result = tool.run(device_id="thermostat_1", action="self_destruct")
    assert result.success is False
    assert "Unsupported action" in result.error


# ---------------------------------------------------------
# Test MockHotelSearchTool, MockFlightSearchTool, MockRestaurantSearchTool
# ---------------------------------------------------------
def test_hotel_search_success():
    tool = MockHotelSearchTool()
    assert tool.name == "hotel_search"
    result = tool.run(destination="Delhi", max_price=3000)
    assert result.success is True
    assert result.output["destination"] == "Delhi"
    assert result.output["count"] > 0


def test_hotel_search_failure_simulation():
    tool = MockHotelSearchTool()
    result = tool.run(destination="Delhi", max_price=3000, fail=True)
    assert result.success is False
    assert "Hotel search service unavailable" in result.error


def test_flight_search_success_and_failure():
    tool = MockFlightSearchTool()
    assert tool.name == "flight_search"
    res = tool.run(destination="Mumbai", origin="Delhi", max_price=5000)
    assert res.success is True
    assert res.output["count"] > 0

    fail_res = tool.run(destination="Mumbai", fail=True)
    assert fail_res.success is False
    assert "Flight search service unavailable" in fail_res.error


def test_restaurant_search_success_and_failure():
    tool = MockRestaurantSearchTool()
    assert tool.name == "restaurant_search"
    res = tool.run(location="Delhi", cuisine="North Indian")
    assert res.success is True
    assert res.output["count"] > 0

    fail_res = tool.run(location="Delhi", fail=True)
    assert fail_res.success is False
    assert "Restaurant search service unavailable" in fail_res.error


# ---------------------------------------------------------
# Test ToolRegistry
# ---------------------------------------------------------
def test_get_mock_tools():
    tools = get_mock_tools()
    assert len(tools) == 9
    names = {t.name for t in tools}
    assert "send_email" in names
    assert "book_ride" in names
    assert "get_weather" in names
    assert "transfer_funds" in names
    assert "create_calendar_event" in names
    assert "control_device" in names
    assert "hotel_search" in names
    assert "flight_search" in names
    assert "restaurant_search" in names


def test_tool_registry_registration_and_lookup():
    registry = ToolRegistry()
    email_tool = MockSendEmailTool()
    registry.register(email_tool)

    assert len(registry) == 1
    assert "send_email" in registry
    assert registry.has_tool("send_email") is True
    assert registry.get("send_email") is email_tool
    assert registry.get_tool("send_email") is email_tool

    # Duplicate registration fails without overwrite=True
    with pytest.raises(ValueError):
        registry.register(MockSendEmailTool())

    # Overwrite succeeds
    new_email_tool = MockSendEmailTool()
    registry.register(new_email_tool, overwrite=True)
    assert registry.get("send_email") is new_email_tool


def test_tool_registry_unregister():
    registry = ToolRegistry()
    tool = MockGetWeatherTool()
    registry.register(tool)

    assert registry.has_tool("get_weather") is True
    unregistered = registry.unregister("get_weather")
    assert unregistered is tool
    assert registry.has_tool("get_weather") is False

    with pytest.raises(KeyError):
        registry.unregister("non_existent_tool")


def test_tool_registry_execution():
    registry = create_default_registry()

    # Successful execution via registry
    result = registry.execute(
        "get_weather",
        location="Tokyo",
        units="celsius",
    )
    assert result.success is True
    assert result.output["location"] == "Tokyo"

    # Execution of unregistered tool fails gracefully
    unregistered_result = registry.execute("unknown_tool_xyz", foo="bar")
    assert unregistered_result.success is False
    assert "not registered" in unregistered_result.error


def test_tool_definitions_export():
    registry = create_default_registry()
    definitions = registry.get_definitions()
    assert len(definitions) == 9

    for defn in definitions:
        assert isinstance(defn, ToolDefinition)
        dump = defn.to_dict()
        assert "name" in dump
        assert "description" in dump
        assert "risk_level" in dump


def test_default_tool_registry_instance():
    assert len(tool_registry) >= 7
    assert "send_email" in tool_registry
    assert "book_ride" in tool_registry
    assert "hotel_search" in tool_registry
