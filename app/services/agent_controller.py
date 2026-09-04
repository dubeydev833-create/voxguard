"""VoxGuard Agent Controller.

Coordinates intent parsing, asynchronous tool dispatch,
session task management, result fencing, and response synthesis.
"""

import asyncio
import re
from typing import Any, Callable, Dict, Optional, Tuple

from app.llm.base import LLMProvider, StructuredIntent
from app.llm.provider import get_llm_provider
from app.models.events import EventType, ToolResultEnvelope
from app.models.state import Session, SessionState
from app.models.tool import ToolResult
from app.services.session_manager import SessionManager, session_manager as default_session_manager
from app.tools.registry import ToolRegistry, tool_registry as default_tool_registry


class AgentController:
    """Core controller coordinating turn lifecycle, LLM intent parsing, and tool execution."""

    def __init__(
        self,
        manager: Optional[SessionManager] = None,
        registry: Optional[ToolRegistry] = None,
        intent_parser: Optional[Callable[[str], Optional[Tuple[str, Dict[str, Any]]]]] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.session_manager = manager or default_session_manager
        self.tool_registry = registry or default_tool_registry
        self.llm_provider = llm_provider or get_llm_provider()
        self.intent_parser = intent_parser or self.default_intent_parser

    def default_intent_parser(self, transcript: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Deterministic intent extractor matching user transcript against available tools."""
        raw = transcript.strip()

        # 1. Flight search intent
        if re.search(r"\b(flights?|fly|plane|airline)\b", raw, re.IGNORECASE):
            from_match = re.search(r"\bfrom\s+([a-zA-Z\s]+?)(?:\s+to\b|\s*$)", raw, re.IGNORECASE)
            to_match = re.search(r"\bto\s+([a-zA-Z\s]+?)(?:\s+(?:from|under|with|for|\?|$))", raw, re.IGNORECASE)
            dest = to_match.group(1).strip() if to_match else "Mumbai"
            origin = from_match.group(1).strip() if from_match else "Delhi"
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else 10000.0
            return "flight_search", {"destination": dest, "origin": origin, "max_price": max_price}

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
            return "restaurant_search", args

        # 3. Hotel search intent
        if re.search(r"\b(hotels?|lodging|stay)\b", raw, re.IGNORECASE) or re.search(r"\bunder\s+\$?\d+", raw, re.IGNORECASE):
            price_match = re.search(r"(?:price|under|max|limit|upto|up to)\s*(?:of|that)?\s*\$?(\d+)", raw, re.IGNORECASE)
            max_price = float(price_match.group(1)) if price_match else 3000.0

            dest_match = re.search(r"\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\s+(?:with|under|max|limit|\?|$))", raw, re.IGNORECASE)
            destination = dest_match.group(1).strip() if dest_match else "Delhi"
            return "hotel_search", {"destination": destination, "max_price": max_price}

        # 4. Book ride intent
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

        # 5. Weather intent
        if re.search(r"\b(weather|temperature|forecast)\b", raw, re.IGNORECASE):
            loc_match = re.search(r"\b(?:weather|forecast|temperature)?\s*\b(?:in|for|at)\s+([a-zA-Z\s]+?)(?:\?|$)", raw, re.IGNORECASE)
            location = loc_match.group(1).strip() if loc_match else "Seattle"
            units = "fahrenheit" if re.search(r"\bfahrenheit\b", raw, re.IGNORECASE) else "celsius"
            return "get_weather", {"location": location, "units": units}

        # 6. Email intent
        if re.search(r"\b(email|mail|send message)\b", raw, re.IGNORECASE):
            email_match = re.search(r"[\w\.-]+@[\w\.-]+", raw)
            recipient = email_match.group(0) if email_match else "recipient@example.com"
            return "send_email", {
                "recipient": recipient,
                "subject": "Voice Assistant Notification",
                "body": raw,
            }

        # 7. Fund transfer intent
        if re.search(r"\b(transfer|wire|send money)\b", raw, re.IGNORECASE):
            amount_match = re.search(r"\$?(\d+(?:\.\d+)?)", raw)
            amount = float(amount_match.group(1)) if amount_match else 100.0
            acc_match = re.search(r"acc[_\w]*", raw, re.IGNORECASE)
            account = acc_match.group(0).upper() if acc_match else "ACC_999999"
            return "transfer_funds", {"recipient_account": account, "amount": amount}

        # 8. Device control intent
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

    def validate_tool_request(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        """Validate tool existence and required arguments.

        Returns error string if invalid, or None if valid.
        """
        if not self.tool_registry.has_tool(tool_name):
            return f"Tool '{tool_name}' is not registered in the tool registry."

        tool = self.tool_registry.get(tool_name)
        if tool and tool.parameters:
            required_params = tool.parameters.get("required", [])
            missing = [p for p in required_params if p not in tool_args or tool_args[p] is None]
            if missing:
                return f"Missing required parameter(s) for tool '{tool_name}': {', '.join(missing)}"

            # Validate common parameter types safely
            if tool_name == "transfer_funds" and "amount" in tool_args:
                try:
                    amt = float(tool_args["amount"])
                    if amt <= 0:
                        return f"Transfer amount must be strictly positive, received: {amt}"
                except (ValueError, TypeError):
                    return f"Amount must be a numeric value, received: {tool_args['amount']}"

            if tool_name in ("hotel_search", "flight_search", "restaurant_search") and "max_price" in tool_args and tool_args["max_price"] is not None:
                try:
                    float(tool_args["max_price"])
                except (ValueError, TypeError):
                    return f"Parameter 'max_price' must be numeric, received: {tool_args['max_price']}"

        return None

    def handle_turn(
        self,
        session_id: str,
        transcript: str,
        simulated_delay: float = 0.0,
        request_id: Optional[str] = None,
    ) -> Session:
        """Process incoming user turn, launch async tool if parsed, and manage fencing.

        Args:
            session_id: Target session ID.
            transcript: User audio transcript or text prompt.
            simulated_delay: Optional simulated delay for asynchronous tool execution.
            request_id: Optional unique request ID. If None, one will be generated.

        Returns:
            The initialized or updated Session object.
        """
        # 1. Start turn via session manager (cancels active task, increments version, sets request_id)
        session = self.session_manager.start_turn(
            session_id=session_id,
            transcript=transcript,
            state=SessionState.THINKING,
            request_id=request_id,
        )
        turn_version = session.current_version
        turn_request_id = session.current_request_id

        # Check if LLM provider has an asynchronous simulated delay
        llm_delay = getattr(self.llm_provider, "delay", 0.0)

        # --- Case 1: Asynchronous LLM Processing (delay > 0) ---
        if llm_delay > 0:
            async def async_turn_worker() -> Optional[str]:
                cur_task = asyncio.current_task()
                try:
                    # 1. Async LLM intent parsing
                    structured_intent = await self.llm_provider.parse_intent(
                        transcript=transcript,
                        session_id=session_id,
                        request_id=turn_request_id,
                        version=turn_version,
                    )
                except asyncio.CancelledError:
                    if cur_task and hasattr(cur_task, "uncancel"):
                        cur_task.uncancel()
                    res = ToolResult.cancelled(
                        tool_name="llm_intent_parsing",
                        message=f"LLM parsing was cancelled for turn {turn_version} ({turn_request_id})",
                        request_id=turn_request_id,
                        version=turn_version,
                    )
                    envelope = ToolResultEnvelope(
                        session_id=session_id,
                        version=turn_version,
                        request_id=turn_request_id,
                        tool_result=res,
                    )
                    self.session_manager.process_tool_result(envelope)
                    return None
                except Exception as exc:
                    # Handle LLM failure safely without crashing backend
                    active_session = self.session_manager.get_session(session_id)
                    if active_session and active_session.current_version == turn_version:
                        active_session.state = SessionState.ERROR
                        fallback_resp = f"I encountered an error interpreting your request: {exc}"
                        active_session.last_response = fallback_resp
                        self.session_manager._emit(
                            EventType.RESPONSE_READY,
                            session_id=session_id,
                            version=turn_version,
                            request_id=turn_request_id,
                            payload={"response": fallback_resp, "error": str(exc), "request_id": turn_request_id},
                        )
                    return None

                # 2. PRE-DISPATCH FENCE CHECK
                # If turn was superseded during LLM processing, NEVER dispatch tool execution!
                active_session = self.session_manager.get_session(session_id)
                if not active_session or active_session.current_version != turn_version or active_session.current_request_id != turn_request_id:
                    res = ToolResult.cancelled(
                        tool_name=structured_intent.intent or "unknown",
                        message=f"Turn superseded during LLM processing (current version {active_session.current_version if active_session else 'none'} != turn {turn_version})",
                        request_id=turn_request_id,
                        version=turn_version,
                    )
                    envelope = ToolResultEnvelope(
                        session_id=session_id,
                        version=turn_version,
                        request_id=turn_request_id,
                        tool_result=res,
                    )
                    self.session_manager.process_tool_result(envelope)
                    return None

                # 3. Direct conversational turn without tool execution
                if not structured_intent.intent:
                    direct_resp = structured_intent.direct_response or f"I received your request: '{transcript}'. How can I assist you further?"
                    active_session.state = SessionState.COMPLETED
                    active_session.committed_version = turn_version
                    active_session.committed_request_id = turn_request_id
                    if turn_request_id and turn_request_id in active_session.requests:
                        active_session.requests[turn_request_id].status = "completed"
                    active_session.last_response = direct_resp
                    active_session.committed_data["response"] = direct_resp
                    self.session_manager._emit(
                        EventType.RESPONSE_READY,
                        session_id=session_id,
                        version=turn_version,
                        request_id=turn_request_id,
                        payload={"response": direct_resp, "request_id": turn_request_id},
                    )
                    return direct_resp

                # 4. Tool Validation
                tool_name = structured_intent.intent
                tool_args = dict(structured_intent.arguments)
                validation_err = self.validate_tool_request(tool_name, tool_args)
                if validation_err:
                    res = ToolResult.fail(
                        tool_name=tool_name,
                        error=validation_err,
                        request_id=turn_request_id,
                        version=turn_version,
                    )
                    envelope = ToolResultEnvelope(
                        session_id=session_id,
                        version=turn_version,
                        request_id=turn_request_id,
                        tool_result=res,
                    )
                    self.session_manager.process_tool_result(envelope)
                    return None

                if simulated_delay > 0 and "delay" not in tool_args:
                    tool_args["delay"] = simulated_delay

                # 5. Transition to TOOL_RUNNING and emit TOOL_STARTED
                self.session_manager.set_state(session_id, SessionState.TOOL_RUNNING)
                self.session_manager._emit(
                    EventType.TOOL_STARTED,
                    session_id=session_id,
                    version=turn_version,
                    request_id=turn_request_id,
                    payload={"tool_name": tool_name, "arguments": tool_args, "request_id": turn_request_id},
                )

                # 6. Execute tool asynchronously
                try:
                    res = await self.tool_registry.execute_async(tool_name, **tool_args)
                    if res.request_id is None:
                        res.request_id = turn_request_id
                    if res.version is None:
                        res.version = turn_version
                except asyncio.CancelledError:
                    if cur_task and hasattr(cur_task, "uncancel"):
                        cur_task.uncancel()
                    res = ToolResult.cancelled(
                        tool_name=tool_name,
                        message=f"Execution of '{tool_name}' was cancelled by a superseding turn or interruption.",
                        request_id=turn_request_id,
                        version=turn_version,
                    )
                except Exception as exc:
                    res = ToolResult.fail(
                        tool_name=tool_name,
                        error=str(exc),
                        request_id=turn_request_id,
                        version=turn_version,
                    )

                # 7. Apply Result Fencing
                envelope = ToolResultEnvelope(
                    session_id=session_id,
                    version=turn_version,
                    request_id=turn_request_id,
                    tool_result=res,
                )
                accepted = self.session_manager.process_tool_result(envelope)

                # 8. Synthesize response if accepted
                if accepted and not res.is_cancelled and res.success:
                    synthesized = self.synthesize_response(tool_name, res)
                    if hasattr(self.llm_provider, "synthesize_response"):
                        try:
                            await self.llm_provider.synthesize_response(
                                tool_name=tool_name,
                                tool_result=res,
                                transcript=transcript,
                                session_id=session_id,
                                request_id=turn_request_id,
                                version=turn_version,
                            )
                        except Exception:
                            pass
                    active_sess = self.session_manager.get_session(session_id)
                    if active_sess:
                        active_sess.last_response = synthesized
                        active_sess.committed_data["response"] = synthesized
                    self.session_manager._emit(
                        EventType.RESPONSE_READY,
                        session_id=session_id,
                        version=turn_version,
                        request_id=turn_request_id,
                        payload={"response": synthesized, "tool_name": tool_name, "request_id": turn_request_id},
                    )
                    return synthesized
                return None

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(async_turn_worker())
                self.session_manager.set_task(session_id, task)
            except RuntimeError:
                pass
            return session

        # --- Case 2: Synchronous / Fast Intent Parsing (delay == 0) ---
        try:
            if hasattr(self.llm_provider, "parse_intent_sync"):
                structured_intent = self.llm_provider.parse_intent_sync(
                    transcript=transcript,
                    session_id=session_id,
                    request_id=turn_request_id,
                    version=turn_version,
                )
            elif self.intent_parser != self.default_intent_parser:
                raw_parsed = self.intent_parser(transcript)
                if raw_parsed:
                    structured_intent = StructuredIntent(intent=raw_parsed[0], arguments=raw_parsed[1])
                else:
                    structured_intent = StructuredIntent(intent=None)
            else:
                raw_parsed = self.default_intent_parser(transcript)
                if raw_parsed:
                    structured_intent = StructuredIntent(intent=raw_parsed[0], arguments=raw_parsed[1])
                else:
                    structured_intent = StructuredIntent(intent=None)
        except Exception as exc:
            session.state = SessionState.ERROR
            fallback_resp = f"I encountered an error interpreting your request: {exc}"
            session.last_response = fallback_resp
            self.session_manager._emit(
                EventType.RESPONSE_READY,
                session_id=session_id,
                version=turn_version,
                request_id=turn_request_id,
                payload={"response": fallback_resp, "error": str(exc), "request_id": turn_request_id},
            )
            return session

        # Direct conversational turn without tool execution
        if not structured_intent.intent:
            direct_response = structured_intent.direct_response or f"I received your request: '{transcript}'. How can I assist you further?"
            session.state = SessionState.COMPLETED
            session.committed_version = turn_version
            session.committed_request_id = turn_request_id
            if turn_request_id and turn_request_id in session.requests:
                session.requests[turn_request_id].status = "completed"
            session.last_response = direct_response
            session.committed_data["response"] = direct_response
            self.session_manager._emit(
                EventType.RESPONSE_READY,
                session_id=session_id,
                version=turn_version,
                request_id=turn_request_id,
                payload={"response": direct_response, "request_id": turn_request_id},
            )
            return session

        # Validate tool request
        tool_name = structured_intent.intent
        tool_args = dict(structured_intent.arguments)
        validation_err = self.validate_tool_request(tool_name, tool_args)
        if validation_err:
            res = ToolResult.fail(
                tool_name=tool_name,
                error=validation_err,
                request_id=turn_request_id,
                version=turn_version,
            )
            envelope = ToolResultEnvelope(
                session_id=session_id,
                version=turn_version,
                request_id=turn_request_id,
                tool_result=res,
            )
            self.session_manager.process_tool_result(envelope)
            return session

        if simulated_delay > 0 and "delay" not in tool_args:
            tool_args["delay"] = simulated_delay

        # Transition to TOOL_RUNNING and emit TOOL_STARTED
        self.session_manager.set_state(session_id, SessionState.TOOL_RUNNING)
        self.session_manager._emit(
            EventType.TOOL_STARTED,
            session_id=session_id,
            version=turn_version,
            request_id=turn_request_id,
            payload={"tool_name": tool_name, "arguments": tool_args, "request_id": turn_request_id},
        )

        # Define background worker for tool execution
        async def tool_worker() -> Optional[str]:
            cur_task = asyncio.current_task()
            try:
                res = await self.tool_registry.execute_async(tool_name, **tool_args)
                if res.request_id is None:
                    res.request_id = turn_request_id
                if res.version is None:
                    res.version = turn_version
            except asyncio.CancelledError:
                if cur_task and hasattr(cur_task, "uncancel"):
                    cur_task.uncancel()
                res = ToolResult.cancelled(
                    tool_name=tool_name,
                    message=f"Execution of '{tool_name}' was cancelled by a superseding turn or interruption.",
                    request_id=turn_request_id,
                    version=turn_version,
                )
            except Exception as exc:
                res = ToolResult.fail(
                    tool_name=tool_name,
                    error=str(exc),
                    request_id=turn_request_id,
                    version=turn_version,
                )

            # Apply Result Fencing
            envelope = ToolResultEnvelope(
                session_id=session_id,
                version=turn_version,
                request_id=turn_request_id,
                tool_result=res,
            )
            accepted = self.session_manager.process_tool_result(envelope)

            if accepted and not res.is_cancelled and res.success:
                synthesized = self.synthesize_response(tool_name, res)
                if hasattr(self.llm_provider, "synthesize_response"):
                    try:
                        await self.llm_provider.synthesize_response(
                            tool_name=tool_name,
                            tool_result=res,
                            transcript=transcript,
                            session_id=session_id,
                            request_id=turn_request_id,
                            version=turn_version,
                        )
                    except Exception:
                        pass
                active_session = self.session_manager.get_session(session_id)
                if active_session:
                    active_session.last_response = synthesized
                    active_session.committed_data["response"] = synthesized
                self.session_manager._emit(
                    EventType.RESPONSE_READY,
                    session_id=session_id,
                    version=turn_version,
                    request_id=turn_request_id,
                    payload={"response": synthesized, "tool_name": tool_name, "request_id": turn_request_id},
                )
                return synthesized
            return None

        # Spawn background task and track it in session_manager
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(tool_worker())
            self.session_manager.set_task(session_id, task)
        except RuntimeError:
            pass

        return session


# Global singleton AgentController
agent_controller = AgentController()
