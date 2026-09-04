"""VoxGuard FastAPI Application Entrypoint.

Registers conversation REST routes, WebSocket event streaming routes,
and system health check endpoints.
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager

from app.api.v1.conversations import router as conversations_router
from app.api.v1.stream import router as stream_router
from app.tools.registry import tool_registry
from app.tools.mock_tools import (
    MockHotelSearchTool,
    MockFlightSearchTool,
    MockRestaurantSearchTool,
)


def register_startup_mock_tools() -> None:
    """Ensure mock tools are registered in default tool registry."""
    for tool_cls in (MockHotelSearchTool, MockFlightSearchTool, MockRestaurantSearchTool):
        tool_instance = tool_cls()
        if not tool_registry.has_tool(tool_instance.name):
            tool_registry.register(tool_instance)


# Register immediately on import
register_startup_mock_tools()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure registered on lifespan startup
    register_startup_mock_tools()
    yield


app = FastAPI(
    title="VoxGuard Voice Agent Guardrail API",
    description="Secure Voice-First AI Agent with Task Tracking and Result Fencing",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(conversations_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Sanitize all unexpected server exceptions to prevent stack trace or secrets leakage."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Static files and Web Dashboard
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/ui", tags=["ui"])
def serve_ui():
    """Interactive VoxGuard Web Client and Voice Dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(status_code=404, content={"detail": "UI dashboard not found."})


@app.get("/health", tags=["health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "ok", "service": "voxguard"}


@app.get("/", tags=["root"])
def root():
    """Root metadata endpoint."""
    return {
        "service": "VoxGuard API",
        "version": "1.0.0",
        "docs_url": "/docs",
    }

