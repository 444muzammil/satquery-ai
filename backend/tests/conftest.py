import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.validator import ValidationResult
from providers.base import RSVLMResponse

@pytest.fixture
def mock_runpod_provider():
    provider = AsyncMock()
    provider.name = "RunPod/EarthDial"
    provider.health_check = AsyncMock(return_value=True)
    provider.get_capabilities = MagicMock(return_value={"is_fallback": False})
    
    mock_response = RSVLMResponse(
        text="Mocked VLM response",
        success=True,
        confidence=85.0,
        model_name="EarthDial",
        is_fallback=False
    )
    provider.analyze = AsyncMock(return_value=mock_response)
    return provider

@pytest.fixture
def mock_fallback_provider():
    provider = AsyncMock()
    provider.name = "Classical CV Fallback"
    provider.health_check = AsyncMock(return_value=True)
    provider.get_capabilities = MagicMock(return_value={"is_fallback": True})
    
    mock_response = RSVLMResponse(
        text="Classical spectral analysis",
        success=True,
        confidence=40.0,
        model_name="Classical CV Fallback",
        is_fallback=True
    )
    provider.analyze = AsyncMock(return_value=mock_response)
    return provider
