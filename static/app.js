/**
 * VoxGuard Interactive Client & Voice Dashboard
 * Connects browser to VoxGuard FastAPI backend, Rime voice synthesis, and telemetry streams.
 */

// State
let currentSessionId = null;
let currentVersion = 0;
let committedVersion = null;
let activeRequestId = null;
let eventSocket = null;
let speechRecognizer = null;
let isRecording = false;

// DOM Elements
const sessionStateBadge = document.getElementById("sessionStateBadge");
const sessionIdVal = document.getElementById("sessionIdVal");
const currVersionVal = document.getElementById("currVersionVal");
const commVersionVal = document.getElementById("commVersionVal");
const activeReqVal = document.getElementById("activeReqVal");
const connectionStatus = document.getElementById("connectionStatus");
const promptInput = document.getElementById("promptInput");
const delayInput = document.getElementById("delayInput");
const speakerSelect = document.getElementById("speakerSelect");
const sendTurnBtn = document.getElementById("sendTurnBtn");
const interruptBtn = document.getElementById("interruptBtn");
const newSessionBtn = document.getElementById("newSessionBtn");
const micBtn = document.getElementById("micBtn");
const micStatusText = document.getElementById("micStatusText");
const micIndicator = document.getElementById("micIndicator");
const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");
const autoTtsCheck = document.getElementById("autoTtsCheck");
const manualTtsBtn = document.getElementById("manualTtsBtn");
const audioStatusBadge = document.getElementById("audioStatusBadge");
const conversationList = document.getElementById("conversationList");
const eventLogList = document.getElementById("eventLogList");
const clearFeedBtn = document.getElementById("clearFeedBtn");

// Tab Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    const tabId = btn.getAttribute("data-tab") + "Tab";
    const content = document.getElementById(tabId);
    if (content) content.classList.add("active");
  });
});

// Clear Log
clearFeedBtn.addEventListener("click", () => {
  eventLogList.innerHTML = '<div class="empty-state">Log cleared. Waiting for events...</div>';
});

// Audio Status Helpers
function setAudioStatus(status, text) {
  audioStatusBadge.textContent = text;
  audioStatusBadge.className = `badge badge-${status}`;
}

// Stop and reset audio immediately on barge-in
function cutOffAudio() {
  if (ttsAudioPlayer) {
    try {
      ttsAudioPlayer.pause();
      ttsAudioPlayer.currentTime = 0;
      ttsAudioPlayer.removeAttribute("src");
      ttsAudioPlayer.load();
    } catch (err) {
      console.warn("Audio stop error:", err);
    }
  }
  setAudioStatus("danger", "Stopped (Barge-in)");
}

// Update State Badges
function updateSessionBadge(state) {
  const stateStr = (state || "IDLE").toLowerCase();
  sessionStateBadge.textContent = (state || "IDLE").toUpperCase();
  sessionStateBadge.className = `badge badge-${stateStr}`;
}

function updateSessionUI(session) {
  if (!session) return;
  currentSessionId = session.session_id;
  currentVersion = session.current_version;
  committedVersion = session.committed_version;
  activeRequestId = session.current_request_id;

  sessionIdVal.textContent = session.session_id;
  currVersionVal.textContent = session.current_version;
  commVersionVal.textContent = session.committed_version !== null && session.committed_version !== undefined ? session.committed_version : "—";
  activeReqVal.textContent = session.current_request_id || "—";
  updateSessionBadge(session.state);
}

// Initialize / Create Session
async function createNewSession() {
  try {
    const res = await fetch("/api/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metadata: { client: "voxguard-web-ui" } }),
    });
    if (!res.ok) throw new Error(`Failed to create session: ${res.statusText}`);
    const data = await res.json();
    updateSessionUI(data);
    appendLogEntry("SYSTEM", `Session initialized: ${data.session_id} (v${data.current_version})`);
    setupWebSocket(data.session_id);
  } catch (err) {
    console.error("Error creating session:", err);
    alert("Could not initialize session with backend.");
  }
}

// WebSocket Setup
function setupWebSocket(sessionId) {
  if (eventSocket) {
    try { eventSocket.close(); } catch (e) {}
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/v1/sessions/${sessionId}/events`;

  connectionStatus.textContent = "WebSocket: Connecting...";
  connectionStatus.className = "badge badge-thinking";

  try {
    eventSocket = new WebSocket(wsUrl);

    eventSocket.onopen = () => {
      connectionStatus.textContent = "WebSocket: Connected";
      connectionStatus.className = "badge badge-connected";
    };

    eventSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleServerEvent(data);
      } catch (err) {
        console.warn("Invalid event JSON:", event.data);
      }
    };

    eventSocket.onclose = () => {
      connectionStatus.textContent = "WebSocket: Disconnected";
      connectionStatus.className = "badge badge-disconnected";
    };

    eventSocket.onerror = (err) => {
      console.warn("WebSocket error:", err);
      connectionStatus.textContent = "WebSocket: Error";
      connectionStatus.className = "badge badge-danger";
    };
  } catch (err) {
    console.error("WebSocket init failed:", err);
  }
}

// Event Log Appender
function appendLogEntry(type, details, extra = null) {
  const empty = eventLogList.querySelector(".empty-state");
  if (empty) empty.remove();

  const entry = document.createElement("div");
  entry.className = "event-entry";

  const top = document.createElement("div");
  top.className = "event-entry-top";

  const typeSpan = document.createElement("span");
  typeSpan.className = `event-type-badge event-type-${type.toLowerCase()}`;
  typeSpan.textContent = `[${type}]`;

  const timeSpan = document.createElement("span");
  timeSpan.className = "meta-label mono";
  timeSpan.textContent = new Date().toLocaleTimeString();

  top.appendChild(typeSpan);
  top.appendChild(timeSpan);
  entry.appendChild(top);

  const body = document.createElement("div");
  body.className = "event-entry-body";
  body.textContent = typeof details === "string" ? details : JSON.stringify(details);
  entry.appendChild(body);

  if (extra) {
    const extraDiv = document.createElement("div");
    extraDiv.className = "event-entry-body mono";
    extraDiv.style.fontSize = "0.72rem";
    extraDiv.style.color = "#a5b4fc";
    extraDiv.textContent = JSON.stringify(extra);
    entry.appendChild(extraDiv);
  }

  eventLogList.appendChild(entry);
  eventLogList.scrollTop = eventLogList.scrollHeight;
}

// Handle Server Events
function handleServerEvent(evt) {
  const eventType = evt.event_type || "UNKNOWN";
  appendLogEntry(eventType, `Version: ${evt.version} | Request: ${evt.request_id || "N/A"}`, evt.payload || null);

  // Update UI version/state if available
  if (evt.version) {
    currentVersion = evt.version;
    currVersionVal.textContent = evt.version;
  }
  if (evt.request_id) {
    activeReqVal.textContent = evt.request_id;
  }

  // Update state badge based on event
  if (eventType === "turn_received") {
    updateSessionBadge("THINKING");
  } else if (eventType === "tool_started") {
    updateSessionBadge("TOOL_EXECUTING");
  } else if (eventType === "turn_interrupted") {
    updateSessionBadge("INTERRUPTED");
    cutOffAudio();
  } else if (eventType === "turn_committed") {
    updateSessionBadge("COMMITTED");
    commVersionVal.textContent = evt.version;
  } else if (eventType === "result_rejected") {
    // Result fence rejection!
    appendLogEntry("FENCE_SHIELD", `🛡️ Result Fence safely rejected stale result for version ${evt.version}!`);
  }
}

// Append Chat Message
function appendChatMessage(role, text, meta = null, isInterrupted = false) {
  const empty = conversationList.querySelector(".empty-state");
  if (empty) empty.remove();

  const msg = document.createElement("div");
  msg.className = `chat-msg chat-${role} ${isInterrupted ? 'interrupted-msg' : ''}`;

  const header = document.createElement("div");
  header.className = "chat-msg-header";
  header.innerHTML = `<span>${role === 'user' ? '👤 User' : '🤖 VoxGuard'}</span><span>${meta || ''}</span>`;
  msg.appendChild(header);

  const content = document.createElement("div");
  content.textContent = text;
  msg.appendChild(content);

  conversationList.appendChild(msg);
  conversationList.scrollTop = conversationList.scrollHeight;
}

// Submit Turn
async function submitTurn() {
  const text = promptInput.value.trim();
  if (!text) {
    alert("Please enter a query or speak into the microphone.");
    return;
  }
  if (!currentSessionId) {
    await createNewSession();
  }

  const delay = parseFloat(delayInput.value) || 0.0;
  promptInput.value = "";
  sendTurnBtn.disabled = true;

  // Immediate UI turn display
  appendChatMessage("user", text, `v${currentVersion + 1}`);
  updateSessionBadge("THINKING");

  try {
    const res = await fetch(`/api/v1/sessions/${currentSessionId}/turns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcript: text,
        simulated_delay: delay,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || res.statusText);
    }

    const session = await res.json();
    updateSessionUI(session);

    if (session.last_response) {
      appendChatMessage("assistant", session.last_response, `v${session.current_version} • committed`);
      if (autoTtsCheck.checked) {
        synthesizeSpeechAudio(session.session_id, session.last_response);
      }
    }
  } catch (err) {
    console.error("Turn submission error:", err);
    appendChatMessage("assistant", `Error: ${err.message}`, null, true);
  } finally {
    sendTurnBtn.disabled = false;
  }
}

// Interrupt Turn (Barge-in)
async function triggerInterrupt() {
  if (!currentSessionId) return;

  // Immediately cut off local audio
  cutOffAudio();
  updateSessionBadge("INTERRUPTED");

  try {
    const res = await fetch(`/api/v1/sessions/${currentSessionId}/interrupt`, {
      method: "POST",
    });
    if (!res.ok) {
      console.warn("Interrupt request failed:", res.statusText);
    } else {
      const data = await res.json();
      updateSessionUI(data);
      appendLogEntry("INTERRUPT", `Barge-in triggered for session ${currentSessionId}`);
    }
  } catch (err) {
    console.error("Interrupt error:", err);
  }
}

// Synthesize Audio via Rime
async function synthesizeSpeechAudio(sessionId, text = null) {
  if (!sessionId) return;
  const speaker = speakerSelect.value || "marsh";
  setAudioStatus("thinking", "Synthesizing audio...");

  try {
    const res = await fetch(`/api/v1/sessions/${sessionId}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker: speaker, text: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      setAudioStatus("danger", "Synthesis rejected");
      appendLogEntry("TTS_WARNING", `TTS synthesis skipped: ${err.detail || res.statusText}`);
      return;
    }

    const ttsData = await res.json();
    if (ttsData.audio_base64) {
      const audioSrc = `data:${ttsData.audio_format || 'audio/wav'};base64,${ttsData.audio_base64}`;
      ttsAudioPlayer.src = audioSrc;
      ttsAudioPlayer.play()
        .then(() => {
          setAudioStatus("connected", `Playing (${ttsData.speaker})`);
        })
        .catch(err => {
          console.warn("Audio autoplay blocked by browser:", err);
          setAudioStatus("muted", "Ready to play (click audio controls)");
        });

      ttsAudioPlayer.onended = () => {
        setAudioStatus("muted", "Playback finished");
      };
    }
  } catch (err) {
    console.error("TTS fetch error:", err);
    setAudioStatus("danger", "Audio error");
  }
}

// Web Speech Recognition (Browser Voice Input)
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.disabled = true;
    micStatusText.textContent = "Voice input unsupported in browser";
    return;
  }

  speechRecognizer = new SpeechRecognition();
  speechRecognizer.continuous = false;
  speechRecognizer.interimResults = true;
  speechRecognizer.lang = "en-US";

  speechRecognizer.onstart = () => {
    isRecording = true;
    micBtn.classList.add("listening");
    micStatusText.textContent = "Listening... Speak now";
    micIndicator.classList.remove("hidden");
  };

  speechRecognizer.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    promptInput.value = transcript;
  };

  speechRecognizer.onerror = (event) => {
    console.warn("Speech recognition error:", event.error);
    stopRecording();
  };

  speechRecognizer.onend = () => {
    stopRecording();
  };
}

function toggleRecording() {
  if (!speechRecognizer) return;
  if (isRecording) {
    speechRecognizer.stop();
  } else {
    // If speaking, user starting to talk acts as a barge-in
    if (ttsAudioPlayer && !ttsAudioPlayer.paused) {
      triggerInterrupt();
    }
    try {
      speechRecognizer.start();
    } catch (e) {
      console.warn("Speech start error:", e);
    }
  }
}

function stopRecording() {
  isRecording = false;
  micBtn.classList.remove("listening");
  micStatusText.textContent = "Start Voice Input";
  micIndicator.classList.add("hidden");
}

// Event Listeners
sendTurnBtn.addEventListener("click", submitTurn);
interruptBtn.addEventListener("click", triggerInterrupt);
newSessionBtn.addEventListener("click", createNewSession);
micBtn.addEventListener("click", toggleRecording);
manualTtsBtn.addEventListener("click", () => {
  if (currentSessionId) {
    synthesizeSpeechAudio(currentSessionId);
  } else {
    alert("No active session.");
  }
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitTurn();
  }
});

// Auto-initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  setupSpeechRecognition();
  createNewSession();
});
