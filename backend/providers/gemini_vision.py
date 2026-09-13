"""
SatQuery AI - Gemini Vision Mock Provider

Temporary provider mapping RS VLM calls to Gemini 2.5 Flash for internal demos.
"""
import httpx
import logging
import time
from typing import List, Optional

from .base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType
from config import Settings

logger = logging.getLogger("satquery.providers.gemini_vision")

class GeminiVisionProvider(RSVLMProvider):
    def __init__(self, settings: Settings):
        self.model_name = "gemini-3.6-flash"
        self.timeout = settings.RS_VLM_TIMEOUT
        self.api_key = getattr(settings, "GEMINI_API_KEY", "")
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
    async def analyze(self, images: List[ImageInput], query: str, task: TaskType) -> RSVLMResponse:
        start_t = time.time()
        
        if not self.api_key:
            return RSVLMResponse(
                text="Gemini API key missing. Please configure GEMINI_API_KEY.",
                is_fallback=False,
                model_name=self.model_name
            )
            
        parts = [{"text": query}]
        
        for img in images:
            b64 = img.clean_base64()
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64
                }
            })
            
        payload = {
            "contents": [
                {
                    "parts": parts
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 300
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                try:
                    answer = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    answer = "Could not extract text from Gemini response."
                
                return RSVLMResponse(
                    text=answer,
                    is_fallback=False,
                    model_name=self.model_name,
                    latency_ms=int((time.time() - start_t) * 1000)
                )
        except Exception as e:
            logger.error(f"Gemini Vision API Error: {e}")
            return RSVLMResponse(
                text=f"API Error: {str(e)}",
                is_fallback=True,
                model_name="Fallback_CV",
                latency_ms=int((time.time() - start_t) * 1000)
            )
    async def health_check(self) -> bool:
        return bool(self.api_key)

    def get_capabilities(self) -> dict:
        return {
            "vqa": True,
            "caption": True,
            "grounding": False
        }

    @property
    def name(self) -> str:
        return "Gemini 3.6 Flash"
