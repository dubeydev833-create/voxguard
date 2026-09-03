"""VoxGuard WebSocket Event Streaming Endpoint.

Provides real-time event streaming for session state transitions,
task cancellations, tool completions, and result fencing rejections.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.events import Event
from app.services.session_manager import session_manager

router = APIRouter(tags=["stream"])


@router.websocket("/sessions/{session_id}/events")
async def stream_session_events(websocket: WebSocket, session_id: str):
    """Stream live events for a specific session via WebSocket.

    Broadcasts:
    - SESSION_CREATED
    - TURN_STARTED
    - INTERRUPTED
    - TOOL_STARTED / TOOL_COMPLETED
    - RESULT_ACCEPTED
    - RESULT_REJECTED_STALE
    - STATE_CHANGED
    """
    await websocket.accept()

    session = session_manager.get_or_create_session(session_id)

    # Send initial connection confirmation
    await websocket.send_json({
        "event_type": "CONNECTED",
        "session_id": session_id,
        "current_version": session.current_version,
        "state": session.state.value,
    })

    # Queue to forward events from session_manager listener to websocket
    queue: asyncio.Queue[Event] = asyncio.Queue()

    def on_event(event: Event):
        if event.session_id == session_id:
            try:
                queue.put_nowait(event)
            except Exception:
                pass

    session_manager.add_listener(on_event)

    async def sender():
        try:
            while True:
                event = await queue.get()
                await websocket.send_text(event.model_dump_json())
                queue.task_done()
        except asyncio.CancelledError:
            pass

    async def receiver():
        try:
            while True:
                # Keep connection alive and listen for client messages / pings
                await websocket.receive_text()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())

    try:
        done, pending = await asyncio.wait(
            [sender_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        session_manager.remove_listener(on_event)
        sender_task.cancel()
        receiver_task.cancel()
