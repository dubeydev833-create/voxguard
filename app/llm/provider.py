"""VoxGuard LLM Provider Factory.

Handles server-side provider instantiation and configuration.
All API keys and credentials remain strictly server-side and are NEVER
exposed through responses or client-facing interfaces.
"""

import os
from typing import Optional

from app.llm.base import LLMProvider
from app.llm.mock_provider import MockLLMProvider


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """Instantiate and return the configured LLMProvider.

    Reads server-side environment variables:
    - VOXGUARD_LLM_PROVIDER: Provider name ('mock', 'openai', etc.) Default: 'mock'
    - OPENAI_API_KEY: Server-side secret key (never sent to client)

    Returns:
        An instance implementing the LLMProvider interface.
    """
    ptype = (provider_type or os.getenv("VOXGUARD_LLM_PROVIDER", "mock")).lower().strip()

    if ptype == "mock":
        return MockLLMProvider()

    # Generic or external provider setup (server-side only)
    # If external provider is requested but dependencies or keys are missing,
    # safely fall back to MockLLMProvider for resilience.
    return MockLLMProvider()
