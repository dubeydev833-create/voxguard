"""VoxGuard Agent Controller.

Coordinates intent parsing, asynchronous tool dispatch,
session task management, result fencing, and response synthesis.
"""

import asyncio
import re
from typing import Any, Callable, Dict, Optional, Tuple

from app.models.events import EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.session_manager import SessionManager, session_manager as default_session_manager
from app.tools.registry import ToolRegistry, tool_registry as default_tool_registry


class AgentController:
    """Core controller coordinating turn lifecycle and tool execution."""

    def __init__(
        self,
        manager: Optional[SessionManager] = None,
        registry: Optional[ToolRegistry] = None,
        intent_parser: Optional[Callable[[str], Optional[Tuple[str, Dict[str, Any]]]]] = None,
    ) -> None:
        self.session_manager = manager or default_session_manager
        self.tool_registry = registry or default_tool_registry
        self.intent_parser = intent_parser or self.default_intent_parser

    def default_intent_parser(self, transcript: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Deterministic intent extractor matching user transcript against available tools."""
        raw = transcript.strip()

        # 1. Hotel search intent
        if re.search(r"\b(hotels?|lodging|stay)\b", raw, re.IGNORECASE) or re.search(r"\bunder\s+\$?\d+", raw, re.IGNORECASE):
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else 3000.0

            dest_match = re.search(r"\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+(?:with|under|max|limit|\?|$))", raw, re.IGNORECASE)
            destination = dest_match.group(1).strip() if dest_match else "Delhi"
            return "hotel_search", {"destination": destination, "max_price": max_price}

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
            return "book_ride", {"pickup_location": pickup, "dropoff_location": dropoff}

        # 3. Weather intent
        if re.search(r"\b(weather|temperature|forecast)\b", raw, re.IGNORECASE):
            loc_match = re.search(r"\b(?:weather|forecast|temperature)?\s*\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$)", raw, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else "Seattle"
            units = "fahrenheit" if re.search(r"\bfahrenheit\b", raw, re.IGNORECASE) else "celsius"
            return "get_weather", {"location": location, "units": units}

        # 4. Email intent
        if re.search(r"\b(email|mail|send message)\b", raw, re.IGNORECASE):
            email_match = re.search(r"[\w\.-]+@[\w\.-]+", raw)
            recipient = email_match.group(0) if email_match else "recipient@example.com"
            return "send_email", {
                "recipient": recipient,
                "subject": "Voice Assistant Notification",
                "body": raw,
            }

        # 5. Fund transfer intent
        if re.search(r"\b(transfer|wire|send money)\b", raw, re.IGNORECASE):
            amount_match = re.search(r"\$?(\d+(?:\.\d+)?)", raw)
            amount = float(amount_match.group(1)) if amount_match else 100.0
            acc_match = re.search(r"acc[_\w]*", raw, re.IGNORECASE)
            account = acc_match.group(0).upper() if acc_match else "ACC_999999"
            return "transfer_funds", {"recipient_account": account, "amount": amount}

        # 6. Device control intent
        if re.search(r"\b(turn on|turn off|lock|unlock|device)\b", raw, re.IGNORECASE):
            action = "turn_on"
            if re.search(r"\bturn off\b", raw, re.IGNORECASE):
                action = "turn_off"
            elif re.search(r"\bunlock\b", raw, re.IGNORECASE):
                action = "unlock"
            elif re.search(r"\block\b", raw, re.IGNORECASE):
                action = "lock"
            return "control_device", {"device_id": "living_room_device", "action": action}

        return None

    def synthesize_response(self, tool_name: str, tool_result: ToolResult) -> str:
        """Formulate a concise natural language response from execution output."""
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
        elif tool_name == "control_device":
            dev = out.get("device_id", "")
            act = out.get("action", "")
            return f"The device '{dev}' action '{act}' was completed successfully."

        return f"Completed {tool_name} successfully."

    def handle_turn(
        self,
        session_id: str,
        transcript: str,
        simulated_delay: float = 0.0,
    ) -> Session:
        """Process incoming user turn, launch async tool if parsed, and manage fencing.

        Args:
            session_id: Target session ID.
            transcript: User audio transcript or text prompt.
            simulated_delay: Optional simulated delay for asynchronous tool execution.

        Returns:
            The initialized or updated Session object.
        """
        # 1. Start turn via session manager (cancels active task, increments version)
        session = self.session_manager.start_turn(
            session_id=session_id,
            transcript=transcript,
            state=SessionState.THINKING,
        )
        turn_version = session.current_version

        # 2. Parse intent
        parsed_intent = self.intent_parser(transcript)

        if not parsed_intent:
            # Direct conversational turn without tool execution
            direct_response = f"I received your request: '{transcript}'. How can I assist you further?"
            session.state = SessionState.COMPLETED
            session.last_response = direct_response
            session.committed_data["response"] = direct_response
            self.session_manager._emit(
                EventType.RESPONSE_READY,
                session_id=session_id,
                version=turn_version,
                payload={"response": direct_response},
            )
            return session

        tool_name, tool_args = parsed_intent

        # 3. Transition to TOOL_RUNNING and emit TOOL_STARTED
        self.session_manager.set_state(session_id, SessionState.TOOL_RUNNING)
        self.session_manager._emit(
            EventType.TOOL_STARTED,
            session_id=session_id,
            version=turn_version,
            payload={"tool_name": tool_name, "arguments": tool_args},
        )

        # 4. Define background worker for tool execution
        async def tool_worker() -> Optional[str]:
            cur_task = asyncio.current_task()
            try:
                if simulated_delay > 0:
                    await asyncio.sleep(simulated_delay)
                res = self.tool_registry.execute(tool_name, **tool_args)
            except asyncio.CancelledError:
                # Interrupted or superseded turn
                if cur_task and hasattr(cur_task, "uncancel"):
                    cur_task.uncancel()
                res = ToolResult.fail(
                    tool_name=tool_name,
                    error=f"Execution of '{tool_name}' was cancelled by a superseding turn or interruption.",
                )
            except Exception as exc:
                res = ToolResult.fail(tool_name=tool_name, error=str(exc))

            # 5. Apply Result Fencing
            envelope = ToolResultEnvelope(
                session_id=session_id,
                version=turn_version,
                tool_result=res,
            )
            accepted = self.session_manager.process_tool_result(envelope)

            if accepted:
                synthesized = self.synthesize_response(tool_name, res)
                active_session = self.session_manager.get_session(session_id)
                if active_session:
                    active_session.last_response = synthesized
                    active_session.committed_data["response"] = synthesized
                self.session_manager._emit(
                    EventType.RESPONSE_READY,
                    session_id=session_id,
                    version=turn_version,
                    payload={"response": synthesized, "tool_name": tool_name},
                )
                return synthesized
            return None

        # 5. Spawn background task and track it in session_manager
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(tool_worker())
            self.session_manager.set_task(session_id, task)
        except RuntimeError:
            pass

        return session


# Global singleton AgentController
agent_controller = AgentController()
