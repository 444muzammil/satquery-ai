"""
SatQuery AI - OpenAI Vision Mock Provider

Temporary provider mapping RS VLM calls to GPT-4o Vision for internal demos.
"""
import httpx
import logging
import time
from typing import List, Optional

from .base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType
from config import Settings

logger = logging.getLogger("satquery.providers.openai_vision")

class OpenAIVisionProvider(RSVLMProvider):
    def __init__(self, settings: Settings):
        self.model_name = "gpt-4o"
        self.timeout = settings.RS_VLM_TIMEOUT
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.api_key = settings.OPENAI_API_KEY
        
    async def analyze(self, images: List[ImageInput], query: str, task: TaskType, config=None) -> RSVLMResponse:
        start_t = time.time()
        
        if not self.api_key:
            return RSVLMResponse(
                text="OpenAI API key missing. Please configure OPENAI_API_KEY.",
                is_fallback=False,
                model_name=self.model_name
            )
            
        content = [{"type": "text", "text": query}]
        
        for img in images:
            b64 = img.clean_base64()
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
            
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 300
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                answer = data['choices'][0]['message']['content']
                
                return RSVLMResponse(
                    text=answer,
                    is_fallback=False,
                    model_name=self.model_name,
                    latency_ms=int((time.time() - start_t) * 1000)
                )
        except Exception as e:
            logger.error(f'OpenAI Vision API Error: {e}')
            return RSVLMResponse(
                text=f'API Error: {str(e)}',
                is_fallback=True,
                model_name='Fallback_CV',
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
        return "GPT-4o Vision"
