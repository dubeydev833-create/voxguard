"""VoxGuard Phase 11 Integration Tests — Frontend/Rime Integration & End-to-End Production Wiring.

Verifies:
1. Rime Voice / TTS synthesis service (offline mock & contract).
2. Guardrail validation preventing interrupted turns from synthesizing audio.
3. FastAPI /api/v1/sessions/{session_id}/tts REST endpoint.
4. Static file serving (/static/...) and Web UI dashboard (/ui).
5. End-to-end flow: User prompt -> Tool -> Result Fence -> Response -> TTS Audio.
6. Interruption (barge-in) flow cleanly invalidates and rejects TTS synthesis.
"""

import asyncio
import base64
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from app.main import app
from app.models.state import Session, SessionState
from app.services.rime_service import RimeService, rime_service
from app.services.session_manager import session_manager


@pytest.fixture
def client():
    """FastAPI synchronous TestClient fixture."""
    return TestClient(app)


# --- 1. Rime Service Unit & Guardrail Tests ---

def test_rime_mock_wav_structure():
    """Verify mock WAV audio generator creates a valid 16-bit PCM WAV container."""
    wav_bytes = RimeService.generate_mock_wav(duration_s=0.2, sample_rate=16000)
    assert len(wav_bytes) > 44  # Standard WAV header is at least 44 bytes
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_rime_service_offline_synthesis():
    """Verify Rime service synthesizes speech payload in offline/test environment."""
    service = RimeService(api_key=None)
    response = await service.synthesize(
        session_id="test-rime-session",
        text="I have booked your hotel in Paris.",
        version=1,
        request_id="req-101",
        speaker="marsh",
    )
    assert response.session_id == "test-rime-session"
    assert response.version == 1
    assert response.request_id == "req-101"
    assert response.text == "I have booked your hotel in Paris."
    assert response.speaker == "marsh"
    assert response.audio_format == "audio/wav"
    assert len(response.audio_base64) > 0

    # Ensure payload decodes to valid WAV bytes
    decoded = base64.b64decode(response.audio_base64)
    assert decoded[:4] == b"RIFF"


def test_rime_can_synthesize_for_session_guardrails():
    """Verify barge-in guardrails prevent synthesis for invalid/interrupted sessions."""
    # 1. Missing session
    allowed, reason = RimeService.can_synthesize_for_session(None)
    assert not allowed
    assert "not found" in reason.lower()

    # 2. Interrupted session (Barge-in protection)
    interrupted_session = Session(
        session_id="s-interrupted",
        current_version=2,
        state=SessionState.INTERRUPTED,
        last_response="Stale response that should not be spoken",
    )
    allowed, reason = RimeService.can_synthesize_for_session(interrupted_session)
    assert not allowed
    assert "interrupted" in reason.lower()

    # 3. Session without response text
    empty_session = Session(
        session_id="s-empty",
        current_version=1,
        state=SessionState.COMPLETED,
        last_response=None,
    )
    allowed, reason = RimeService.can_synthesize_for_session(empty_session)
    assert not allowed
    assert "no synthesized text" in reason.lower()

    # 4. Valid session
    valid_session = Session(
        session_id="s-valid",
        current_version=1,
        state=SessionState.COMPLETED,
        last_response="Your flight is confirmed.",
    )
    allowed, reason = RimeService.can_synthesize_for_session(valid_session)
    assert allowed
    assert reason is None


# --- 2. FastAPI UI and Static Endpoints Tests ---

def test_ui_endpoint_serves_html_dashboard(client):
    """Verify GET /ui serves the interactive VoxGuard dashboard."""
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "VoxGuard" in resp.text
    assert "Result Fencing" in resp.text
    assert "ttsAudioPlayer" in resp.text


def test_static_assets_served(client):
    """Verify /static/style.css and /static/app.js are served correctly."""
    css_resp = client.get("/static/style.css")
    assert css_resp.status_code == 200
    assert ":root" in css_resp.text

    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "createNewSession" in js_resp.text
    assert "cutOffAudio" in js_resp.text


# --- 3. TTS API Endpoint Tests ---

def test_tts_endpoint_missing_session(client):
    """Verify POST /tts returns 404 for unknown session."""
    resp = client.post("/api/v1/sessions/unknown-id-xyz/tts", json={})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_tts_endpoint_interrupted_session_rejected(client):
    """Verify POST /tts returns 400 when session has been interrupted (barge-in)."""
    create_resp = client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["session_id"]

    client.post(f"/api/v1/sessions/{session_id}/interrupt")

    tts_resp = client.post(f"/api/v1/sessions/{session_id}/tts", json={})
    assert tts_resp.status_code == 400
    assert "interrupted" in tts_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_tts_endpoint_success_flow():
    """Verify POST /tts successfully returns audio for a completed conversational turn."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Create session
        create_resp = await ac.post("/api/v1/sessions", json={})
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # 2. Submit conversational turn (direct response)
        turn_resp = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "Hello, how can you help me today?"},
        )
        assert turn_resp.status_code == 200
        await asyncio.sleep(0.02)

        # Inspect session
        sess_resp = await ac.get(f"/api/v1/sessions/{session_id}")
        assert sess_resp.status_code == 200
        sess_data = sess_resp.json()
        assert sess_data["state"] == "COMPLETED"
        assert sess_data["last_response"] is not None

        # 3. Request voice audio synthesis via Rime
        tts_resp = await ac.post(
            f"/api/v1/sessions/{session_id}/tts",
            json={"speaker": "amber"},
        )
        assert tts_resp.status_code == 200
        tts_data = tts_resp.json()
        assert tts_data["session_id"] == session_id
        assert tts_data["speaker"] == "amber"
        assert tts_data["audio_format"] == "audio/wav"
        assert len(tts_data["audio_base64"]) > 0


# --- 4. End-to-End Voice / Interruption Wiring Tests ---

@pytest.mark.asyncio
async def test_e2e_turn_completion_and_voice_pipeline():
    """Full End-to-End test: User query -> Tool -> ResultFence -> Response -> Rime Voice."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Create session
        create_res = await ac.post("/api/v1/sessions", json={"metadata": {"test": "phase11_e2e"}})
        session_id = create_res.json()["session_id"]

        # 2. Submit turn (weather tool execution with 0.0s delay)
        turn_res = await ac.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"transcript": "What is the weather in Tokyo?", "simulated_delay": 0.0},
        )
        assert turn_res.status_code == 200
        await asyncio.sleep(0.05)

        sess_res = await ac.get(f"/api/v1/sessions/{session_id}")
        sess_data = sess_res.json()
        assert sess_data["state"] == "COMPLETED"
        assert sess_data["committed_version"] == 1
        assert "Tokyo" in sess_data["last_response"]

        # 3. Synthesize voice audio
        tts_res = await ac.post(
            f"/api/v1/sessions/{session_id}/tts",
            json={"speaker": "marsh"},
        )
        assert tts_res.status_code == 200
        tts_data = tts_res.json()
        assert tts_data["version"] == 1
        assert tts_data["text"] == sess_data["last_response"]
        assert len(tts_data["audio_base64"]) > 50


def test_e2e_barge_in_prevents_stale_voice_synthesis(client):
    """Test that barge-in interrupts session and prevents voice playback for stale turn."""
    # 1. Create session
    create_res = client.post("/api/v1/sessions", json={})
    session_id = create_res.json()["session_id"]

    # 2. Simulate barge-in interruption before turn completion
    client.post(f"/api/v1/sessions/{session_id}/interrupt")

    # 3. Voice synthesis must be rejected
    tts_res = client.post(f"/api/v1/sessions/{session_id}/tts", json={})
    assert tts_res.status_code == 400
    assert "cannot synthesize" in tts_res.json()["detail"].lower()
