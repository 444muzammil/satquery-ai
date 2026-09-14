"""
SatQuery AI — Centralized Configuration

All environment variables and application settings in one place.
RunPod plug-and-play: set RS_VLM_PROVIDER=runpod + endpoint + key to activate.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("satquery")


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    # --- Agent Controller (text-only orchestration) ---
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # --- Primary RS VLM (EarthDial on RunPod) ---
    RS_VLM_PROVIDER: str = "fallback"          # "runpod" | "local" | "fallback" | "openai" | "gemini"
    RS_VLM_ENDPOINT: str = ""                  # RunPod serverless/pod URL
    RS_VLM_API_KEY: str = ""                   # RunPod API key
    RS_VLM_MODEL: str = "EarthDial_4B"         # Model/checkpoint identifier
    RS_VLM_TIMEOUT: int = 60                   # Request timeout in seconds
    RS_VLM_MAX_RETRIES: int = 3                # Max retry attempts

    # --- Optional secondary VLM (e.g., GeoChat) ---
    SECONDARY_VLM_ENDPOINT: str = ""
    SECONDARY_VLM_API_KEY: str = ""
    SECONDARY_VLM_MODEL: str = ""

    # --- Specialist models ---
    CHANGE_MODEL_ENDPOINT: str = ""
    CHANGE_MODEL_API_KEY: str = ""

    FUSION_MODEL_ENDPOINT: str = ""
    FUSION_MODEL_API_KEY: str = ""

    GROUNDING_MODEL_ENDPOINT: str = ""
    GROUNDING_MODEL_API_KEY: str = ""

    # --- Runtime ---
    RS_VLM_LOCAL: bool = False                 # Load model locally (requires GPU + transformers)
    LOG_LEVEL: str = "INFO"
    MAX_IMAGE_SIZE_MB: int = 100
    ALLOWED_ORIGINS: list = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ])

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        # Explicitly load .env file to ensure variables are available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not installed, relying on system environment variables.")

        s = cls()
        s.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        s.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

        s.RS_VLM_PROVIDER = os.getenv("RS_VLM_PROVIDER", "fallback")
        s.RS_VLM_ENDPOINT = os.getenv("RS_VLM_ENDPOINT", "")
        s.RS_VLM_API_KEY = os.getenv("RS_VLM_API_KEY", "")
        s.RS_VLM_MODEL = os.getenv("RS_VLM_MODEL", "EarthDial_4B")
        s.RS_VLM_TIMEOUT = int(os.getenv("RS_VLM_TIMEOUT", "60"))
        s.RS_VLM_MAX_RETRIES = int(os.getenv("RS_VLM_MAX_RETRIES", "3"))

        s.SECONDARY_VLM_ENDPOINT = os.getenv("SECONDARY_VLM_ENDPOINT", "")
        s.SECONDARY_VLM_API_KEY = os.getenv("SECONDARY_VLM_API_KEY", "")
        s.SECONDARY_VLM_MODEL = os.getenv("SECONDARY_VLM_MODEL", "")

        s.CHANGE_MODEL_ENDPOINT = os.getenv("CHANGE_MODEL_ENDPOINT", "")
        s.CHANGE_MODEL_API_KEY = os.getenv("CHANGE_MODEL_API_KEY", "")
        s.FUSION_MODEL_ENDPOINT = os.getenv("FUSION_MODEL_ENDPOINT", "")
        s.FUSION_MODEL_API_KEY = os.getenv("FUSION_MODEL_API_KEY", "")
        s.GROUNDING_MODEL_ENDPOINT = os.getenv("GROUNDING_MODEL_ENDPOINT", "")
        s.GROUNDING_MODEL_API_KEY = os.getenv("GROUNDING_MODEL_API_KEY", "")

        s.RS_VLM_LOCAL = os.getenv("RS_VLM_LOCAL", "false").lower() in ("true", "1", "yes")
        s.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        s.MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "100"))

        origins_env = os.getenv("ALLOWED_ORIGINS", "")
        if origins_env:
            s.ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]

        return s

    def is_rs_vlm_available(self) -> bool:
        """Check if a real RS VLM provider is configured (not fallback)."""
        if self.RS_VLM_PROVIDER == "runpod":
            return bool(self.RS_VLM_ENDPOINT)
        if self.RS_VLM_PROVIDER == "local":
            return self.RS_VLM_LOCAL
        return False

    def is_specialist_available(self, specialist: str) -> bool:
        """Check if a specialist model endpoint is configured."""
        mapping = {
            "change": self.CHANGE_MODEL_ENDPOINT,
            "fusion": self.FUSION_MODEL_ENDPOINT,
            "grounding": self.GROUNDING_MODEL_ENDPOINT,
            "secondary_vlm": self.SECONDARY_VLM_ENDPOINT,
        }
        return bool(mapping.get(specialist, ""))


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Singleton settings instance
settings = Settings.from_env()
setup_logging(settings.LOG_LEVEL)
