import time
import logging
import asyncio
from typing import Any, Dict, List, Optional

import httpx
import requests

from providers.base import RSVLMProvider, RSVLMResponse, ImageInput, TaskType, Modality
from config import Settings

logger = logging.getLogger("satquery.providers.runpod")

MODEL_REGISTRY = {
    "EarthDial_4B": {
        "tasks": ["vqa", "caption", "grounding"],
        "modalities": ["rgb"],
        "max_images": 1
    },
    "EarthDial_Change": {
        "tasks": ["change_detection", "change_vqa"],
        "modalities": ["rgb"],
        "max_images": 2
    },
    "EarthDial_Fusion": {
        "tasks": ["fusion"],
        "modalities": ["rgb", "sar"],
        "max_images": 2
    }
}

class RunPodProvider(RSVLMProvider):
    """
    RunPod Provider for EarthDial and other RS VLMs hosted on RunPod serverless/pods.
    """
    
    def __init__(self, settings: Settings):
        self.endpoint = settings.RS_VLM_ENDPOINT
        self.api_key = settings.RS_VLM_API_KEY
        self.model = settings.RS_VLM_MODEL
        self.timeout = settings.RS_VLM_TIMEOUT
        self.max_retries = settings.RS_VLM_MAX_RETRIES

    @property
    def name(self) -> str:
        return f"RunPod/{self.model}"

    def get_capabilities(self) -> Dict[str, Any]:
        return MODEL_REGISTRY.get(self.model, {
            "tasks": ["vqa", "caption"],
            "modalities": ["rgb"],
            "max_images": 1
        })

    async def health_check(self) -> bool:
        if not self.endpoint:
            return False
            
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        url = f"{self.endpoint}/health" if not self.endpoint.endswith("/runsync") else self.endpoint.replace("/runsync", "/health")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    logger.debug(f"Health check GET {url} OK")
                    return True
        except Exception as e:
            logger.debug(f"Health check GET failed: {e}")
            
        try:
            payload = {"input": {"ping": True}}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
                if resp.status_code in (200, 400, 422):
                    logger.debug(f"Health check POST {self.endpoint} OK (status {resp.status_code})")
                    return True
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            
        return False

    async def analyze(
        self,
        images: List[ImageInput],
        query: str,
        task: TaskType,
        config: Optional[Dict[str, Any]] = None,
    ) -> RSVLMResponse:
        start_time = time.time()
        
        try:
            input_data = {
                "query": query,
                "task": task.value,
                "model": self.model
            }
            
            for img in images:
                if img.role == "primary":
                    input_data["image"] = img.clean_base64()
                elif img.role == "temporal":
                    input_data["image_b"] = img.clean_base64()
                elif img.role == "sar" or img.modality == Modality.SAR:
                    input_data["sar_image"] = img.clean_base64()
                else:
                    if "image" not in input_data:
                        input_data["image"] = img.clean_base64()
                    elif "image_b" not in input_data:
                        input_data["image_b"] = img.clean_base64()

            if config:
                input_data.update(config)

            payload = {"input": input_data}
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            logger.debug(f"[{self.name}] Sending request to {self.endpoint} for task={task.value}")
            
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                        resp = await client.post(self.endpoint, json=payload, headers=headers)
                        
                    if resp.status_code in (429, 502, 503):
                        if attempt < self.max_retries:
                            backoff = 1.5 ** attempt
                            logger.warning(f"[{self.name}] HTTP {resp.status_code}. Retrying in {backoff:.2f}s...")
                            await asyncio.sleep(backoff)
                            attempt += 1
                            continue
                        else:
                            return RSVLMResponse(
                                text="",
                                model_name=self.name,
                                latency_ms=self._measure_latency(start_time),
                                error=f"HTTP {resp.status_code} after {self.max_retries} retries"
                            )
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    logger.debug(f"[{self.name}] Received response")
                    
                    text = ""
                    if "output" in data and isinstance(data["output"], dict):
                        output = data["output"]
                        text = output.get("answer") or output.get("generated_text") or ""
                    else:
                        text = data.get("answer") or data.get("generated_text") or data.get("text") or ""
                        
                    return RSVLMResponse(
                        text=text or str(data),
                        model_name=self.name,
                        latency_ms=self._measure_latency(start_time),
                        raw_response=data
                    )

                except httpx.TimeoutException:
                    if attempt == 0 and self.max_retries > 0:
                        logger.warning(f"[{self.name}] Timeout. Retrying once...")
                        attempt += 1
                        continue
                    else:
                        return RSVLMResponse(
                            text="",
                            model_name=self.name,
                            latency_ms=self._measure_latency(start_time),
                            error="Request timed out"
                        )
                except httpx.NetworkError as e:
                    if attempt < self.max_retries:
                        backoff = 1.5 ** attempt
                        logger.warning(f"[{self.name}] Network error: {e}. Retrying in {backoff:.2f}s...")
                        await asyncio.sleep(backoff)
                        attempt += 1
                        continue
                    else:
                        return RSVLMResponse(
                            text="",
                            model_name=self.name,
                            latency_ms=self._measure_latency(start_time),
                            error=f"Network error: {str(e)}"
                        )
                except httpx.HTTPStatusError as e:
                    return RSVLMResponse(
                        text="",
                        model_name=self.name,
                        latency_ms=self._measure_latency(start_time),
                        error=f"HTTP error {e.response.status_code}: {e.response.text}"
                    )
                
            return RSVLMResponse(
                text="",
                model_name=self.name,
                latency_ms=self._measure_latency(start_time),
                error="Exceeded max retries"
            )

        except Exception as e:
            logger.exception(f"[{self.name}] Unexpected error during analyze")
            return RSVLMResponse(
                text="",
                model_name=self.name,
                latency_ms=self._measure_latency(start_time),
                error=f"Unexpected error: {str(e)}"
            )
