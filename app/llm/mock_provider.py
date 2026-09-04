"""VoxGuard Mock LLM Provider.

Provides deterministic, offline intent extraction and response synthesis for tests
and local development without external API key dependencies.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional

from app.llm.base import LLMProvider, StructuredIntent
from app.models.tool import ToolResult


class MockLLMProvider(LLMProvider):
    """Deterministic LLM Provider with controllable delay, failure simulation, and call recording."""

    def __init__(
        self,
        delay: float = 0.0,
        fail: bool = False,
        override_intent: Optional[StructuredIntent] = None,
        mock_responses: Optional[Dict[str, str]] = None,
    ) -> None:
        self.delay = delay
        self.fail = fail
        self.override_intent = override_intent
        self.mock_responses = mock_responses or {}
        self.parse_calls: List[Dict[str, Any]] = []
        self.synthesize_calls: List[Dict[str, Any]] = []

    def _parse_regex(self, transcript: str) -> StructuredIntent:
        """Internal regex extraction for mock structured intents."""
        raw = transcript.strip()

        # 1. Flight search intent
        if re.search(r"\b(flights?|fly|plane|airline)\b", raw, re.IGNORECASE):
            from_match = re.search(r"\bfrom\s+([a-zA-Z\s]+?)(?:\s+to\b|\s*$)", raw, re.IGNORECASE)
            to_match = re.search(r"\bto\s+([a-zA-Z\s]+?)(?:\s+(?:from|under|with|for|\?|$))", raw, re.IGNORECASE)
            dest = to_match.group(1).strip() if to_match else "Mumbai"
            origin = from_match.group(1).strip() if from_match else "Delhi"
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else 10000.0
            return StructuredIntent(
                intent="flight_search",
                arguments={"destination": dest, "origin": origin, "max_price": max_price},
            )

        # 2. Restaurant search intent
        if re.search(r"\b(restaurants?|dining|food|eat|cafe|bistro)\b", raw, re.IGNORECASE):
            loc_match = re.search(r"\b(?:in|for|at|near)\s+([a-zA-Z\s]+?)(?:\s+(?:with|under|max|limit|\?|$))", raw, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else "Delhi"
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else None
            cuisine_match = re.search(r"\b(north indian|south indian|chinese|italian|mexican|mughlai|thai|continental)\b", raw, re.IGNORECASE)
            cuisine = cuisine_match.group(1).title() if cuisine_match else "North Indian"
            args: Dict[str, Any] = {"location": location, "cuisine": cuisine}
            if max_price is not None:
                args["max_price"] = max_price
            return StructuredIntent(
                intent="restaurant_search",
                arguments=args,
            )

        # 3. Hotel search intent
        if re.search(r"\b(hotels?|lodging|stay)\b", raw, re.IGNORECASE) or re.search(r"\bunder\s+\$?\d+", raw, re.IGNORECASE):
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else 3000.0

            dest_match = re.search(r"\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+(?:with|under|max|limit|\?|$))", raw, re.IGNORECASE)
            destination = dest_match.group(1).strip() if dest_match else "Delhi"
            return StructuredIntent(
                intent="hotel_search",
                arguments={"destination": destination, "max_price": max_price},
            )

        # 2. Book ride intent
        if re.search(r"\b(ride|cab|uber|taxi)\b", raw, re.IGNORECASE):
            pickup = "Current Location"
            dropoff = "Downtown"
            from_match = re.search(r"\bfrom\s+([^,]+?)(?:\s+to\b|\s*$)", raw, re.IGNORECASE)
            to_match = re.search(r"\bto\s+([^,]+?)(?:\s+from\b|\s*$)", raw, re.IGNORECASE)
            if from_match:
                pickup = from_match.group(1).strip()
            if to_match:
                dropoff = to_match.group(1).strip()
            return StructuredIntent(
                intent="book_ride",
                arguments={"pickup_location": pickup, "dropoff_location": dropoff},
            )

        # 3. Weather intent
        if re.search(r"\b(weather|temperature|forecast)\b", raw, re.IGNORECASE):
            loc_match = re.search(r"\b(?:weather|forecast|temperature)?\s*\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$)", raw, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else "Seattle"
            units = "fahrenheit" if re.search(r"\bfahrenheit\b", raw, re.IGNORECASE) else "celsius"
            return StructuredIntent(
                intent="get_weather",
                arguments={"location": location, "units": units},
            )

        # 4. Email intent
        if re.search(r"\b(email|mail|send message)\b", raw, re.IGNORECASE):
            email_match = re.search(r"[\w\.-]+@[\w\.-]+", raw)
            recipient = email_match.group(0) if email_match else "recipient@example.com"
            return StructuredIntent(
                intent="send_email",
                arguments={
                    "recipient": recipient,
                    "subject": "Voice Assistant Notification",
                    "body": raw,
                },
            )

        # 5. Fund transfer intent
        if re.search(r"\b(transfer|wire|send money)\b", raw, re.IGNORECASE):
            amount_match = re.search(r"\$?(\d+(?:\.\d+)?)", raw)
            amount = float(amount_match.group(1)) if amount_match else 100.0
            acc_match = re.search(r"acc[_\w]*", raw, re.IGNORECASE)
            account = acc_match.group(0).upper() if acc_match else "ACC_999999"
            return StructuredIntent(
                intent="transfer_funds",
                arguments={"recipient_account": account, "amount": amount},
            )

        # 6. Device control intent
        if re.search(r"\b(turn on|turn off|lock|unlock|device)\b", raw, re.IGNORECASE):
            action = "turn_on"
            if re.search(r"\bturn off\b", raw, re.IGNORECASE):
                action = "turn_off"
            elif re.search(r"\bunlock\b", raw, re.IGNORECASE):
                action = "unlock"
            elif re.search(r"\block\b", raw, re.IGNORECASE):
                action = "lock"
            return StructuredIntent(
                intent="control_device",
                arguments={"device_id": "living_room_device", "action": action},
            )

        # Default conversational intent (no tool invocation)
        return StructuredIntent(
            intent=None,
            arguments={},
            direct_response=f"I received your request: '{transcript}'. How can I assist you further?",
        )

    def parse_intent_sync(
        self,
        transcript: str,
        session_id: str = "",
        request_id: str = "",
        version: int = 1,
    ) -> StructuredIntent:
        """Synchronously extract structured intent and record call."""
        self.parse_calls.append({
            "transcript": transcript,
            "session_id": session_id,
            "request_id": request_id,
            "version": version,
        })

        if self.fail:
            raise RuntimeError("Mock LLM Provider simulated failure")

        if self.override_intent is not None:
            return self.override_intent

        return self._parse_regex(transcript)

    async def parse_intent(
        self,
        transcript: str,
        session_id: str,
        request_id: str,
        version: int,
    ) -> StructuredIntent:
        """Extract structured intent with optional simulated delay or failure."""
        if self.delay > 0:
            self.parse_calls.append({
                "transcript": transcript,
                "session_id": session_id,
                "request_id": request_id,
                "version": version,
            })
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("Mock LLM Provider simulated failure")
            if self.override_intent is not None:
                return self.override_intent
            return self._parse_regex(transcript)

        return self.parse_intent_sync(transcript, session_id, request_id, version)

    async def synthesize_response(
        self,
        tool_name: str,
        tool_result: ToolResult,
        transcript: str,
        session_id: str,
        request_id: str,
        version: int,
    ) -> str:
        """Synthesize natural language response from tool output."""
        self.synthesize_calls.append({
            "tool_name": tool_name,
            "tool_result": tool_result,
            "transcript": transcript,
            "session_id": session_id,
            "request_id": request_id,
            "version": version,
        })

        if tool_name in self.mock_responses:
            return self.mock_responses[tool_name]

        if not tool_result.success:
            return f"I encountered an error executing {tool_name}: {tool_result.error}"

        out = tool_result.output or {}
        if tool_name == "hotel_search":
            hotels = out.get("hotels", [])
            count = out.get("count", len(hotels))
            dest = out.get("destination", "Delhi")
            max_p = out.get("max_price", "")
            currency_sym = "₹" if "delhi" in str(dest).lower() else "$"
            price_str = f"{int(max_p)}" if isinstance(max_p, (int, float)) else str(max_p)
            hotel_names = [f"{h['name']} ({currency_sym}{h['price']})" for h in hotels[:2]]
            detail = f": {', '.join(hotel_names)}" if hotel_names else ""
            return f"I found {count} hotels in {dest} under {currency_sym}{price_str}{detail}."
        elif tool_name == "book_ride":
            ride_id = out.get("ride_id", "confirmed")
            fare = out.get("estimated_fare", 18.50)
            return f"Your ride ({ride_id}) is confirmed! Estimated fare: ${fare:.2f}."
        elif tool_name == "get_weather":
            loc = out.get("location", "the area")
            temp = out.get("temperature", "")
            cond = out.get("condition", "")
            unit = "°C" if out.get("units") == "celsius" else "°F"
            return f"The weather in {loc} is currently {temp}{unit} and {cond}."
        elif tool_name == "send_email":
            rec = out.get("recipient", "")
            return f"Your email to {rec} has been successfully sent."
        elif tool_name == "transfer_funds":
            amt = out.get("amount", 0)
            acc = out.get("recipient_account", "")
            return f"Successfully transferred ${amt:.2f} to account {acc}."
        elif tool_name == "flight_search":
            flights = out.get("flights", [])
            count = out.get("count", len(flights))
            dest = out.get("destination", "destination")
            f_names = [f"{f['flight']} (₹{f['price']})" for f in flights[:2] if isinstance(f, dict) and "flight" in f]
            detail = f": {', '.join(f_names)}" if f_names else ""
            return f"I found {count} flights to {dest}{detail}."
        elif tool_name == "restaurant_search":
            restaurants = out.get("restaurants", [])
            count = out.get("count", len(restaurants))
            loc = out.get("location", "Delhi")
            r_names = [r.get("name", "") for r in restaurants[:2] if isinstance(r, dict) and "name" in r]
            detail = f": {', '.join(r_names)}" if r_names else ""
            return f"I found {count} restaurants in {loc}{detail}."
        elif tool_name == "control_device":
            dev = out.get("device_id", "")
            act = out.get("action", "")
            return f"The device '{dev}' action '{act}' was completed successfully."

        return f"Completed {tool_name} successfully."
