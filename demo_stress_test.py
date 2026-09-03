"""VoxGuard Live Interactive Demo Stress Test.

Demonstrates real-time voice agent session lifecycle, mid-flight interruption,
task cancellation, and Result Fencing protection against stale tool completion.
"""

import asyncio
import json
import sys
import time
from typing import Any, Dict, List
import httpx
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_HTTP_URL = "http://127.0.0.1:8000"
BASE_WS_URL = "ws://127.0.0.1:8000"

# ANSI Color Codes for clean visual trace
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"


def print_banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{CYAN}{BOLD}{line}")
    print(f"  {title}")
    print(f"{line}{RESET}\n")


async def main():
    print_banner("VOXGUARD LIVE INTERACTIVE STRESS TEST & RESULT FENCING DEMO")

    collected_events: List[Dict[str, Any]] = []
    listener_stop_event = asyncio.Event()

    async with httpx.AsyncClient(base_url=BASE_HTTP_URL) as client:
        # -------------------------------------------------------------
        # STEP 1: Create a new session via POST /api/v1/sessions
        # -------------------------------------------------------------
        print(f"{BOLD}[1/5] Creating new VoxGuard session...{RESET}")
        create_resp = await client.post("/api/v1/sessions", json={"metadata": {"channel": "demo_cli"}})
        if create_resp.status_code != 201:
            print(f"{RED}Failed to create session: {create_resp.status_code} {create_resp.text}{RESET}")
            sys.exit(1)

        session_data = create_resp.json()
        session_id = session_data["session_id"]
        print(f"      {GREEN}Session created successfully!{RESET}")
        print(f"      Session ID : {BOLD}{session_id}{RESET}")
        print(f"      State      : {session_data['state']}")
        print(f"      Version    : {session_data['current_version']}\n")

        # -------------------------------------------------------------
        # STEP 2: Connect WebSocket telemetry stream
        # -------------------------------------------------------------
        ws_url = f"{BASE_WS_URL}/api/v1/sessions/{session_id}/events"
        print(f"{BOLD}[2/5] Connecting to WebSocket telemetry stream: {CYAN}{ws_url}{RESET}...")

        async with websockets.connect(ws_url) as ws:
            # Read initial handshake
            init_msg = json.loads(await ws.recv())
            print(f"      {GREEN}WebSocket connected! Initial event: {init_msg['event_type']}{RESET}\n")

            # Background listener to display and record real-time telemetry
            async def event_stream_listener():
                try:
                    while not listener_stop_event.is_set():
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        event = json.loads(raw_msg)
                        collected_events.append(event)

                        evt_type = event.get("event_type", "UNKNOWN")
                        version = event.get("version", 0)
                        payload = event.get("payload", {})

                        # Color-coded log format
                        if evt_type == "TURN_STARTED":
                            tag = f"{BLUE}[TURN_STARTED]{RESET}"
                            detail = f"Version: {version} | Prompt: \"{payload.get('transcript', '')}\""
                        elif evt_type == "TOOL_STARTED":
                            tag = f"{MAGENTA}[TOOL_STARTED]{RESET}"
                            detail = f"Version: {version} | Tool: {payload.get('tool_name')} | Args: {payload.get('arguments')}"
                        elif evt_type == "CANCELLATION_REQUESTED":
                            tag = f"{YELLOW}[CANCELLATION_REQUESTED]{RESET}"
                            detail = f"Version: {version} | Reason: {payload.get('reason')}"
                        elif evt_type == "RESULT_REJECTED_STALE":
                            tag = f"{RED}{BOLD}[RESULT_REJECTED_STALE]{RESET}"
                            detail = f"FENCE BLOCKED: Version {payload.get('result_version')} rejected! Active session version is {payload.get('session_current_version')}."
                        elif evt_type == "RESULT_ACCEPTED":
                            tag = f"{GREEN}{BOLD}[RESULT_ACCEPTED]{RESET}"
                            detail = f"FENCE APPROVED: Version {version} result accepted for {payload.get('tool_name')}!"
                        elif evt_type == "RESPONSE_READY":
                            tag = f"{GREEN}[RESPONSE_READY]{RESET}"
                            detail = f"Synthesized Response: \"{payload.get('response')}\""
                        elif evt_type == "INTERRUPTED":
                            tag = f"{YELLOW}[INTERRUPTED]{RESET}"
                            detail = f"Version: {version} | State: {payload.get('state')}"
                        else:
                            tag = f"[EVENT:{evt_type}]"
                            detail = str(payload)

                        ts = time.strftime("%H:%M:%S", time.localtime(event.get("timestamp", time.time())))
                        print(f"      {CYAN}{ts}{RESET} {tag} {detail}")
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass

            listener_task = asyncio.create_task(event_stream_listener())

            # -------------------------------------------------------------
            # STEP 3: Dispatch Turn 1 (V1) with 5s mock tool delay
            # -------------------------------------------------------------
            print(f"{BOLD}[3/5] Dispatching Turn 1 (V1): 'Find hotels in Delhi under 5000' (5s delay)...{RESET}")
            t1_resp = await client.post(
                f"/api/v1/sessions/{session_id}/turns",
                json={
                    "transcript": "Find hotels in Delhi under 5000",
                    "simulated_delay": 5.0,
                },
            )
            print(f"      HTTP Response: {t1_resp.status_code} | Turn Version: {t1_resp.json()['current_version']}")

            # -------------------------------------------------------------
            # STEP 4: Wait 2.0 seconds, then send V2 mid-flight interruption
            # -------------------------------------------------------------
            print(f"\n{YELLOW}      --> Simulating mid-flight barge-in: waiting 2.0s before user speaks...{RESET}")
            await asyncio.sleep(2.0)

            print(f"\n{BOLD}[4/5] User interrupts mid-flight! Dispatching Turn 2 (V2): 'Actually make that under 3000'...{RESET}")
            t2_resp = await client.post(
                f"/api/v1/sessions/{session_id}/turns",
                json={
                    "transcript": "Actually make that under 3000",
                    "simulated_delay": 0.0,
                },
            )
            print(f"      HTTP Response: {t2_resp.status_code} | Turn Version: {t2_resp.json()['current_version']}")

            # -------------------------------------------------------------
            # STEP 5: Wait for V1 to attempt completion and get rejected
            # -------------------------------------------------------------
            print(f"\n{BOLD}[5/5] Listening for Result Fencing decisions on live telemetry stream...{RESET}")
            # Wait 4 seconds to allow V1's 5s task to wake up, finish, and hit the fence
            await asyncio.sleep(4.0)

            # Signal listener to stop
            listener_stop_event.set()
            await listener_task

        # Query final committed session state
        final_resp = await client.get(f"/api/v1/sessions/{session_id}")
        final_data = final_resp.json()

        print_banner("VERIFICATION SUMMARY")
        event_types = [e["event_type"] for e in collected_events]

        cancellation_verified = "CANCELLATION_REQUESTED" in event_types or "INTERRUPTED" in event_types
        stale_rejected_verified = "RESULT_REJECTED_STALE" in event_types
        accepted_verified = "RESULT_ACCEPTED" in event_types
        response_ready_verified = "RESPONSE_READY" in event_types

        print(f"  [{GREEN if 'TURN_STARTED' in event_types else RED}X{RESET}] V1 turn initiation detected (version 1)")
        print(f"  [{GREEN if cancellation_verified else RED}X{RESET}] Mid-flight interruption & cancellation detected for V1")
        print(f"  [{GREEN if stale_rejected_verified else RED}X{RESET}] Delayed V1 result rejected by fence: RESULT_REJECTED_STALE")
        print(f"  [{GREEN if accepted_verified else RED}X{RESET}] V2 result accepted by fence: RESULT_ACCEPTED")
        print(f"  [{GREEN if response_ready_verified else RED}X{RESET}] Natural language response synthesized: RESPONSE_READY\n")

        print(f"{BOLD}Final Committed Session State:{RESET}")
        print(f"  - Session ID        : {final_data['session_id']}")
        print(f"  - Current Version   : {final_data['current_version']}")
        print(f"  - State             : {GREEN}{final_data['state']}{RESET}")
        print(f"  - Committed Version : {GREEN}{final_data['committed_version']}{RESET}")
        print(f"  - Committed MaxPrice: {GREEN}₹{final_data['committed_data'].get('max_price')}{RESET} (V1's ₹5000 was discarded!)")
        print(f"  - Final Spoken Text : {CYAN}\"{final_data.get('last_response')}\"{RESET}\n")

        if stale_rejected_verified and accepted_verified and final_data["committed_version"] == 2:
            print(f"{GREEN}{BOLD}>>> RESULT FENCING DEMO PASSED PERFECTLY! <<<{RESET}\n")
        else:
            print(f"{RED}{BOLD}>>> DEMO CHECKS FAILED <<< {RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
