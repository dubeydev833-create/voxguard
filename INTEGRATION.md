# VoxGuard Backend API Integration Contract

This document specifies the backend API contract for all clients and callers communicating with the VoxGuard runtime.

> **Security Note**:
> No API keys, LLM tokens, or provider secrets are required or accepted from clients. All LLM providers, model credentials, and external tool secrets reside strictly server-side in backend environment variables. Callers only interact with session IDs and request IDs.

---

## 1. Core Architectural Guarantees

1. **Request Identity (`request_id`)**:
   - Every conversational turn and tool execution is bound to a `request_id`.
   - Callers may supply an explicit `request_id` in the request payload. If omitted, the backend generates a unique identifier (format: `req_<hex10>`).
   - `request_id` is tracked on the session (`current_request_id`), propagated through LLM interpretation and tool execution, emitted in telemetry events, and recorded in `committed_request_id` once results are committed.

2. **Monotonic Versioning (`current_version`)**:
   - Sessions initialize at version `0`.
   - Every new conversational turn submitted to `/turns` monotonically increments `current_version` ($V_n \to V_{n+1}$).
   - The version represents the exact conversational epoch.

3. **Asynchronous Cancellation**:
   - Initiating a new turn while a prior turn or tool is executing supersedes the previous turn and initiates cancellation of its active background `asyncio.Task`.
   - Calling `/interrupt` explicitly cancels active background tasks and transitions the session state to `INTERRUPTED`.

4. **Result Fencing**:
   - Results from asynchronous tools pass through the server-side `ResultFence` before they can modify session state.
   - Invariant: **Only results matching the session's active `current_version` (and active `current_request_id`) are accepted (`RESULT_ACCEPTED`).**
   - Any late-completing, interrupted, or obsolete tool results are rejected (`RESULT_REJECTED_STALE`), discarded, and guaranteed never to mutate conversation state or trigger natural-language response generation.

---

## 2. API Endpoints Reference

| Method | Path | Description | Success Status |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/sessions` | Create or initialize a session | `201 Created` |
| `GET` | `/api/v1/sessions/{session_id}` | Retrieve session state and version metadata | `200 OK` |
| `GET` | `/api/v1/sessions/{session_id}/events` | Retrieve chronological recorded events | `200 OK` |
| `POST` | `/api/v1/sessions/{session_id}/turns` | Submit user transcript (starts/supersedes turn) | `200 OK` |
| `POST` | `/api/v1/sessions/{session_id}/interrupt` | Explicitly interrupt and cancel active turn | `200 OK` |
| `POST` | `/api/v1/sessions/{session_id}/results` | Submit raw tool result to Result Fence pipeline | `200 OK` |
| `POST` | `/api/v1/sessions/{session_id}/tts` | Synthesize voice audio via Rime for latest turn | `200 OK` |
| `GET` | `/ui` | Interactive Web Dashboard and Voice Client UI | `200 OK` |
| `WS` | `/api/v1/sessions/{session_id}/events` | Real-time bidirectional WebSocket event stream | `101 Switching Protocols` |


---

## 3. Request & Response Schemas

### 3.1 `POST /api/v1/sessions`
Initializes a new session or registers a client-provided session identifier.

**Request Body (Optional)**:
```json
{
  "session_id": "sess_custom_id_123",
  "metadata": {
    "client": "web_v1"
  }
}
```
*If `session_id` is omitted, the backend generates one formatted as `sess_<hex10>`.*

**Response Body (`201 Created`)**:
```json
{
  "session_id": "sess_custom_id_123",
  "current_version": 0,
  "current_request_id": null,
  "committed_request_id": null,
  "state": "IDLE",
  "last_transcript": null,
  "committed_version": null,
  "committed_data": {},
  "last_response": null,
  "last_event": "SESSION_CREATED",
  "created_at": 1788457461.52,
  "updated_at": 1788457461.52
}
```

---

### 3.2 `GET /api/v1/sessions/{session_id}`
Retrieves current state, versioning, request identity, and latest event status.

**Response Body (`200 OK`)**:
```json
{
  "session_id": "sess_custom_id_123",
  "current_version": 2,
  "current_request_id": "req_turn_002",
  "committed_request_id": "req_turn_002",
  "state": "COMPLETED",
  "last_transcript": "Find hotels in Downtown under 3000",
  "committed_version": 2,
  "committed_data": {
    "hotel_search": { "hotels": [...] }
  },
  "last_response": "I found 2 hotels in Downtown under 3000.",
  "last_event": "RESPONSE_READY",
  "created_at": 1788457461.52,
  "updated_at": 1788457470.05
}
```

---

### 3.3 `GET /api/v1/sessions/{session_id}/events`
Inspects recorded historical telemetry events in chronological order.

**Query Parameters**:
- `event_type` (Optional, string): Filter by specific event type (e.g. `RESULT_REJECTED_STALE`, `TOOL_COMPLETED`).

**Response Body (`200 OK`)**:
```json
[
  {
    "event_id": "evt_1a2b3c4d5e",
    "session_id": "sess_custom_id_123",
    "event_type": "TURN_STARTED",
    "version": 1,
    "request_id": "req_turn_001",
    "timestamp": 1788457461.55,
    "payload": {
      "transcript": "Find hotels in Downtown under 5000"
    }
  },
  {
    "event_id": "evt_9f8e7d6c5b",
    "session_id": "sess_custom_id_123",
    "event_type": "RESULT_REJECTED_STALE",
    "version": 1,
    "request_id": "req_turn_001",
    "timestamp": 1788457465.10,
    "payload": {
      "reason": "Stale result version",
      "result_version": 1,
      "session_current_version": 2,
      "tool_name": "hotel_search"
    }
  }
]
```

---

### 3.4 `POST /api/v1/sessions/{session_id}/turns`
Starts a new conversational turn or supersedes an ongoing turn.

**Request Body**:
```json
{
  "transcript": "Find hotels in Downtown under 3000",
  "request_id": "req_optional_custom_id",
  "state": "THINKING",
  "simulated_delay": null
}
```
*Field Details*:
- `transcript` (string, required, min_length=1): Natural language user input. Empty strings or whitespace-only inputs return HTTP `422`.
- `request_id` (string, optional): Client-assigned request tracker. Auto-generated if omitted.
- `state` (string, optional): Optional override of initial state (default: `THINKING`).
- `simulated_delay` (float, optional, ge=0.0): Optional delay in seconds for testing.

**Response Body (`200 OK`)**:
Returns `SessionResponse` reflecting the incremented `current_version` and new `current_request_id`.

---

### 3.5 `POST /api/v1/sessions/{session_id}/interrupt`
Explicitly halts active background execution for the session.

**Request Body**: None.

**Behavior**:
- Cancels active background `asyncio.Task`.
- Updates `session.state` to `INTERRUPTED`.
- Emits `INTERRUPTED` event.
- Stale background results will be rejected by the Result Fence if they complete late.

**Response Body (`200 OK`)**:
Returns `SessionResponse` with `state: "INTERRUPTED"`.

---

### 3.6 `POST /api/v1/sessions/{session_id}/results`
Allows direct submission of an asynchronous tool execution result to the Result Fence validation engine.

**Request Body**:
```json
{
  "version": 2,
  "request_id": "req_turn_002",
  "call_id": "call_abc123",
  "tool_result": {
    "tool_name": "hotel_search",
    "success": true,
    "result": {
      "hotels": [{"name": "Grand Hotel", "price": 2500}]
    },
    "error": null,
    "execution_time": 0.35,
    "request_id": "req_turn_002",
    "version": 2
  },
  "metadata": {}
}
*Field Details*:
- `version` (integer, required, ge=0): Monotonic version epoch associated with the tool result. Negative values return HTTP `422`.
- `tool_result` (ToolResult, required): The execution output envelope.
- `request_id` (string, optional): Request identity associated with the tool result.

**Response Body (`200 OK`)**:
```json
{
  "accepted": true,
  "session_id": "sess_custom_id_123",
  "current_session_version": 2,
  "submitted_version": 2,
  "request_id": "req_turn_002"
}
```
*(If rejected as stale, `"accepted": false` is returned; session state remains unchanged).*

---

### 3.7 `POST /api/v1/sessions/{session_id}/tts`
Synthesizes speech audio via Rime for the latest committed natural-language response.

**Guarantees**:
- Only committed responses for non-interrupted sessions can be synthesized.
- If the session is currently in `INTERRUPTED` state or has no response text, returns `400 Bad Request`.
- All Rime API credentials remain strictly server-side; client receives standard base64-encoded audio.

**Request Body (Optional)**:
```json
{
  "speaker": "marsh",
  "text": null
}
```

**Response Body (`200 OK`)**:
```json
{
  "session_id": "sess_custom_id_123",
  "version": 2,
  "request_id": "req_turn_002",
  "text": "I found 2 hotels in Downtown under 3000.",
  "audio_base64": "UklGRi4AAABXQVZFZm10IBAAAA...",
  "audio_format": "audio/wav",
  "speaker": "marsh",
  "execution_time": 0.012
}
```

---

### 3.8 `WS /api/v1/sessions/{session_id}/events`
Live WebSocket connection delivering real-time event notifications.

**Connection Flow**:
1. Client opens: `ws://<host>:<port>/api/v1/sessions/{session_id}/events`
2. Server immediately sends initial handshake frame:
   ```json
   {
     "event_type": "CONNECTED",
     "session_id": "sess_custom_id_123",
     "current_version": 2,
     "state": "IDLE"
   }
   ```
3. Chronological runtime events are streamed as JSON objects conforming to the Event schema:
   ```json
   {
     "event_id": "evt_3a1b4c9e8f",
     "event_type": "TOOL_STARTED",
     "session_id": "sess_custom_id_123",
     "version": 2,
     "request_id": "req_turn_002",
     "timestamp": 1788457470.12,
     "payload": {
       "tool_name": "hotel_search",
       "arguments": {"location": "Downtown", "max_price": 3000}
     }
   }
   ```

---

## 4. Error Responses

All API errors return standardized JSON envelopes.

### 4.1 404 Not Found
Returned when an operation targets a non-existent `session_id`.
```json
{
  "detail": "Session 'sess_unknown' not found."
}
```

### 4.2 422 Unprocessable Entity
Returned when request parameters fail schema validation (e.g. empty transcript).
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "transcript"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": { "min_length": 1 }
    }
  ]
}
```

### 4.3 500 Internal Server Error
Returned on unexpected server-side execution failures. Internal stack traces and provider keys are never included in error details.
```json
{
  "detail": "Internal server error"
}
```

---

## 5. Summary of Event Types

| Event Type | Description |
| :--- | :--- |
| `SESSION_CREATED` | New session created and initialized |
| `TURN_STARTED` | Turn initiated; `request_id` and new `version` established |
| `TOOL_STARTED` | Asynchronous tool task launched |
| `TOOL_COMPLETED` | Asynchronous tool completed execution |
| `TOOL_FAILED` | Tool execution encountered an exception |
| `TOOL_CANCELLED` | Tool execution task was cancelled |
| `RESULT_ACCEPTED` | Tool result validated by Result Fence and committed |
| `RESULT_REJECTED_STALE` | Tool result rejected by Result Fence due to version/request mismatch |
| `RESPONSE_READY` | Natural-language response generated for user presentation |
| `INTERRUPTED` | Turn was explicitly or implicitly interrupted |
| `STATE_CHANGED` | Session lifecycle state transitioned |

---

## 6. Reliability & Production Guardrails

1. **Cancellation is Best-Effort**:
   - Cancellation (`task.cancel()`) is an optimization to reclaim execution resources, never the sole mechanism for safety.
   - If a tool catches, ignores, or raises during cancellation, or completes late, the runtime remains fully resilient.
   - **Result Fence is the authoritative correctness mechanism**: regardless of task timing, stale results cannot alter conversation state or trigger voice synthesis.

2. **Session Isolation**:
   - Every session is strictly isolated with independent state, version numbers, and reentrant locks.
   - Concurrent turns or interruptions in Session A can never mutate or read Session B's version, request ID, or committed state.

3. **Exception Isolation & Failure Handling**:
   - Tool execution failures are semantically distinct (`TOOL_FAILED`) from cancellations (`TOOL_CANCELLED`).
   - Exceptions inside tools do not crash the session manager or FastAPI server.
   - New conversational turns can be initiated immediately following a tool failure to recover the session.
   - HTTP 500 responses are sanitized (`{"detail": "Internal server error"}`) to prevent leaking stack traces or internal secrets.

4. **Task Lifecycle & Memory Management**:
   - Active `asyncio.Task` references are automatically cleared via completion callbacks upon normal return, failure, or cancellation, preventing memory leaks.

5. **Observability & Telemetry Hygiene**:
   - Complete event history is available via WebSocket and `GET /api/v1/sessions/{session_id}/events`.
   - Event payloads and session state contain only sanitized business data—never credentials, tokens, or raw secrets.

---

## 7. LLM Intent Parsing & Tool Registry

VoxGuard features a server-side LLM intent parser and an extensible `ToolRegistry`:

1. **Deterministic Mock LLM Provider**:
   - Built for offline testing, local development, and reliable unit verification.
   - Extracts structured intent (`tool_name`, `arguments`) from conversational speech without external network dependencies.
   - Fully supports controllable simulated delay, failure simulation, and intent overrides.

2. **Standard Tool Registry**:
   - **`hotel_search`**: Searches hotel accommodations by destination and maximum price.
   - **`restaurant_search`**: Discovers dining options filtered by location, cuisine, or budget limit.
   - **`flight_search`**: Queries flight options matching origin, destination, and maximum fare.
   - **`book_ride`**, **`get_weather`**, **`send_email`**, **`transfer_funds`**, **`create_calendar_event`**, **`control_device`**: Operational domain tools.
   - Supports failure simulation (`fail=True` / `simulate_failure=True`) and latency simulation (`delay=N`).

