"""VoxGuard Mock Tools.

Mock implementations of common voice agent tools for development,
guardrail verification, and integration testing.
"""

import asyncio
from typing import Any, Dict, List, Optional
import uuid

from app.models.tool import RiskLevel, ToolResult
from app.tools.base import BaseTool


class MockSendEmailTool(BaseTool):
    """Mock tool simulating sending an email message."""

    name: str = "send_email"
    description: str = "Send an email message to a specified recipient."
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_confirmation: bool = True
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient.",
            },
            "subject": {
                "type": "string",
                "description": "Subject of the email.",
            },
            "body": {
                "type": "string",
                "description": "Body content of the email.",
            },
            "cc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of CC email addresses.",
            },
        },
        "required": ["recipient", "subject", "body"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        recipient = kwargs["recipient"]
        subject = kwargs["subject"]
        body = kwargs["body"]
        cc = kwargs.get("cc", [])

        if "@" not in recipient:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Invalid email address: '{recipient}'",
            )

        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        return ToolResult.ok(
            tool_name=self.name,
            output={
                "message_id": message_id,
                "status": "sent",
                "recipient": recipient,
                "subject": subject,
                "body_preview": body[:50] + ("..." if len(body) > 50 else ""),
                "cc": cc,
            },
        )


class MockBookRideTool(BaseTool):
    """Mock tool simulating booking a transportation ride."""

    name: str = "book_ride"
    description: str = "Book a transportation ride from pickup to dropoff destination."
    risk_level: RiskLevel = RiskLevel.HIGH
    requires_confirmation: bool = True
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "pickup_location": {
                "type": "string",
                "description": "Starting address or landmark.",
            },
            "dropoff_location": {
                "type": "string",
                "description": "Destination address or landmark.",
            },
            "ride_type": {
                "type": "string",
                "enum": ["standard", "premium", "xl"],
                "default": "standard",
                "description": "Category of the ride.",
            },
        },
        "required": ["pickup_location", "dropoff_location"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        pickup = kwargs["pickup_location"]
        dropoff = kwargs["dropoff_location"]
        ride_type = kwargs.get("ride_type", "standard")

        ride_id = f"ride_{uuid.uuid4().hex[:8]}"
        fare_estimates = {"standard": 18.50, "premium": 32.00, "xl": 28.00}
        fare = fare_estimates.get(ride_type, 18.50)

        return ToolResult.ok(
            tool_name=self.name,
            output={
                "ride_id": ride_id,
                "status": "confirmed",
                "pickup_location": pickup,
                "dropoff_location": dropoff,
                "ride_type": ride_type,
                "estimated_fare": fare,
                "eta_minutes": 6,
                "driver": {"name": "Alex M.", "vehicle": "Toyota Camry", "plate": "7XYZ89"},
            },
        )


class MockGetWeatherTool(BaseTool):
    """Mock tool simulating querying weather conditions."""

    name: str = "get_weather"
    description: str = "Get the current weather forecast for a specified location."
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or geographical location name.",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "Temperature measurement unit.",
            },
        },
        "required": ["location"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        location = kwargs["location"]
        units = kwargs.get("units", "celsius")

        temp = 21 if units == "celsius" else 70

        return ToolResult.ok(
            tool_name=self.name,
            output={
                "location": location,
                "temperature": temp,
                "units": units,
                "condition": "Partly Cloudy",
                "humidity": 52,
                "wind_speed_kmh": 14,
            },
        )


class MockTransferFundsTool(BaseTool):
    """Mock tool simulating a financial fund transfer."""

    name: str = "transfer_funds"
    description: str = "Transfer monetary funds to a destination account."
    risk_level: RiskLevel = RiskLevel.CRITICAL
    requires_confirmation: bool = True
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "recipient_account": {
                "type": "string",
                "description": "Identifier or account number of recipient.",
            },
            "amount": {
                "type": "number",
                "description": "Monetary amount to transfer.",
            },
            "currency": {
                "type": "string",
                "default": "USD",
                "description": "Currency code (e.g. USD, EUR).",
            },
        },
        "required": ["recipient_account", "amount"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        recipient_account = kwargs["recipient_account"]
        amount = kwargs["amount"]
        currency = kwargs.get("currency", "USD")

        try:
            amount_num = float(amount)
        except (ValueError, TypeError):
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Amount must be a numeric value, received: {amount}",
            )

        if amount_num <= 0:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Transfer amount must be strictly positive, received: {amount_num}",
            )

        tx_id = f"tx_{uuid.uuid4().hex[:10]}"
        return ToolResult.ok(
            tool_name=self.name,
            output={
                "transaction_id": tx_id,
                "status": "completed",
                "recipient_account": recipient_account,
                "amount": amount_num,
                "currency": currency,
            },
        )


class MockCalendarTool(BaseTool):
    """Mock tool simulating calendar event scheduling."""

    name: str = "create_calendar_event"
    description: str = "Create a scheduled event on the user's calendar."
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the calendar event.",
            },
            "start_time": {
                "type": "string",
                "description": "ISO 8601 formatted start time.",
            },
            "end_time": {
                "type": "string",
                "description": "ISO 8601 formatted end time.",
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses.",
            },
        },
        "required": ["title", "start_time", "end_time"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        title = kwargs["title"]
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        attendees = kwargs.get("attendees", [])

        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        return ToolResult.ok(
            tool_name=self.name,
            output={
                "event_id": event_id,
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "attendees": attendees,
                "status": "scheduled",
            },
        )


class MockDeviceControlTool(BaseTool):
    """Mock tool simulating smart home device control."""

    name: str = "control_device"
    description: str = "Control a smart home IoT device (lights, lock, thermostat)."
    risk_level: RiskLevel = RiskLevel.HIGH
    requires_confirmation: bool = True
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "device_id": {
                "type": "string",
                "description": "Identifier of the target device.",
            },
            "action": {
                "type": "string",
                "enum": ["turn_on", "turn_off", "lock", "unlock", "set_temperature"],
                "description": "Action to perform on the device.",
            },
            "value": {
                "type": "string",
                "description": "Optional parameter value (e.g. temperature).",
            },
        },
        "required": ["device_id", "action"],
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        device_id = kwargs["device_id"]
        action = kwargs["action"]
        value = kwargs.get("value")

        valid_actions = ["turn_on", "turn_off", "lock", "unlock", "set_temperature"]
        if action not in valid_actions:
            return ToolResult.fail(
                tool_name=self.name,
                error=f"Unsupported action '{action}'. Must be one of: {valid_actions}",
            )

        return ToolResult.ok(
            tool_name=self.name,
            output={
                "device_id": device_id,
                "action": action,
                "value": value,
                "status": "success",
            },
        )


class MockHotelSearchTool(BaseTool):
    """Mock tool simulating hotel search with price and destination filtering."""

    name: str = "hotel_search"
    description: str = "Search for hotels within a price limit and destination."
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "max_price": {
                "type": "number",
                "description": "Maximum price per night.",
            },
            "destination": {
                "type": "string",
                "default": "Downtown",
                "description": "Destination city or neighborhood.",
            },
            "delay": {
                "type": "number",
                "default": 0.0,
                "description": "Simulated latency delay in seconds.",
            },
        },
        "required": ["max_price"],
    }

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        if kwargs.get("fail") or kwargs.get("simulate_failure"):
            return ToolResult.fail(
                tool_name=self.name,
                error=kwargs.get("error_message", "Hotel search service unavailable"),
            )
        max_price = float(kwargs["max_price"])
        destination = kwargs.get("destination", "Delhi")

        hotel_database = [
            {"name": "The Oberoi New Delhi", "price": 4800, "stars": 5},
            {"name": "Connaught Comfort Hotel", "price": 2800, "stars": 4},
            {"name": "Delhi Heritage Residency", "price": 1900, "stars": 3},
            {"name": "Paharganj Budget Inn", "price": 1200, "stars": 3},
        ]
        matching_hotels = [h for h in hotel_database if h["price"] <= max_price]

        return ToolResult.ok(
            tool_name=self.name,
            output={
                "destination": destination,
                "max_price": max_price,
                "hotels": matching_hotels,
                "count": len(matching_hotels),
                "status": "found",
            },
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        delay = float(kwargs.get("delay", kwargs.get("simulated_delay", 0.0)) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        return self.execute_sync(**kwargs)


class MockFlightSearchTool(BaseTool):
    """Mock tool simulating flight search."""

    name: str = "flight_search"
    description: str = "Search for flights to a destination within a budget."
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Arrival airport or city."},
            "origin": {"type": "string", "default": "Delhi", "description": "Departure airport or city."},
            "max_price": {"type": "number", "description": "Maximum fare in INR."},
            "delay": {"type": "number", "default": 0.0, "description": "Simulated latency delay in seconds."},
        },
        "required": ["destination"],
    }

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        if kwargs.get("fail") or kwargs.get("simulate_failure"):
            return ToolResult.fail(
                tool_name=self.name,
                error=kwargs.get("error_message", "Flight search service unavailable"),
            )
        dest = kwargs["destination"]
        origin = kwargs.get("origin", "Delhi")
        max_p = float(kwargs.get("max_price", 10000))
        flights = [
            {"flight": "AI-801", "origin": origin, "destination": dest, "price": 4500},
            {"flight": "6E-202", "origin": origin, "destination": dest, "price": 3200},
        ]
        matching = [f for f in flights if f["price"] <= max_p]
        return ToolResult.ok(
            tool_name=self.name,
            output={"flights": matching, "count": len(matching), "destination": dest, "origin": origin, "status": "found"},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        delay = float(kwargs.get("delay", kwargs.get("simulated_delay", 0.0)) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        return self.execute_sync(**kwargs)


class MockRestaurantSearchTool(BaseTool):
    """Mock tool simulating restaurant search."""

    name: str = "restaurant_search"
    description: str = "Search for dining restaurants by location, cuisine, or price."
    risk_level: RiskLevel = RiskLevel.LOW
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "default": "Delhi"},
            "destination": {"type": "string", "description": "Alias for location"},
            "cuisine": {"type": "string", "default": "North Indian"},
            "max_price": {"type": "number"},
            "delay": {"type": "number", "default": 0.0, "description": "Simulated latency delay in seconds."},
        },
        "required": [],
    }

    def execute_sync(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        if kwargs.get("fail") or kwargs.get("simulate_failure"):
            return ToolResult.fail(
                tool_name=self.name,
                error=kwargs.get("error_message", "Restaurant search service unavailable"),
            )
        loc = kwargs.get("location") or kwargs.get("destination") or "Delhi"
        cuisine = kwargs.get("cuisine", "North Indian")
        restaurants = [
            {"name": "Bukhara", "cuisine": "North Indian", "avg_cost": 2500},
            {"name": "Karim's", "cuisine": "Mughlai", "avg_cost": 900},
            {"name": "Saravana Bhavan", "cuisine": "South Indian", "avg_cost": 500},
        ]
        max_p = kwargs.get("max_price")
        if max_p is not None:
            restaurants = [r for r in restaurants if r["avg_cost"] <= float(max_p)]
        return ToolResult.ok(
            tool_name=self.name,
            output={"restaurants": restaurants, "count": len(restaurants), "location": loc, "cuisine": cuisine, "status": "found"},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.validate_arguments(**kwargs)
        delay = float(kwargs.get("delay", kwargs.get("simulated_delay", 0.0)) or 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        return self.execute_sync(**kwargs)


def get_mock_tools() -> List[BaseTool]:
    """Return an instantiated list of all standard mock tools."""
    return [
        MockSendEmailTool(),
        MockBookRideTool(),
        MockGetWeatherTool(),
        MockTransferFundsTool(),
        MockCalendarTool(),
        MockDeviceControlTool(),
        MockHotelSearchTool(),
        MockFlightSearchTool(),
        MockRestaurantSearchTool(),
    ]
