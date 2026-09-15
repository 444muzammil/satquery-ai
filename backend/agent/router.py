import os
import re
import json
import time
import requests
from typing import Optional, Tuple, List
from config import Settings

def call_llm_orchestrator(prompt: str, json_mode: bool = False, api_key: str = "") -> Optional[str]:
    if not api_key:
        return None
        
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            elif resp.status_code in [429, 502, 503]:
                time.sleep(1.5 ** attempt)
                continue
        except Exception:
            pass
    return None

def _orchestrate_with_llm(request, api_key: str) -> Optional[dict]:
    intent_prompt = f"""You are the Master Agentic Orchestrator for SatQuery AI (SIH26167 Remote Sensing Assistant).
Deconstruct the user's natural-language query into an execution plan.

USER QUERY: "{request.query}"
AVAILABLE INPUTS:
- Image A (Base Telemetry): True
- Image B (Temporal Telemetry): {request.has_bitemporal}
- SAR Sensor Telemetry: {request.has_sar}

TAXONOMY OF PERMITTED TOOLS:
- RS_VQA: Answering questions about specific features, counts, or characteristics.
- RS_CAPTION: Generating holistic scene descriptions or terrain summaries.
- RS_GROUNDING: Localizing visual features using bounding boundaries. Target MUST be open-vocabulary.
- SPATIAL_COMPARATOR: Required for queries comparing density across regions (e.g., 'most developed', 'where is there more development', 'highest concentration of X'). MUST run after RS_GROUNDING.
- AREA_CALCULATOR: Calculating spatial coverage percentages or ground extent. MUST run after RS_GROUNDING.
- CHANGE_DETECTION: Comparing bi-temporal imagery to determine surface variations.
- OPTICAL_SAR_FUSION: Joint multi-modal reasoning using optical context and SAR radar backscatter.

RULES:
1. Preserve open-vocabulary targets in 'target'. Do NOT force into rigid classes.
2. If the user asks for comparative density, sequence: RS_GROUNDING -> SPATIAL_COMPARATOR.
3. If the user asks 'how much' or 'area', sequence: RS_GROUNDING -> AREA_CALCULATOR.
4. If the user mentions "SAR", "radar", "backscatter", or asks to combine/fuse imagery, you MUST use OPTICAL_SAR_FUSION.
5. If the user mentions "change", "before", "after", or "difference", use CHANGE_DETECTION.
6. CRITICAL: If SAR Sensor Telemetry is True, you MUST substitute RS_GROUNDING with OPTICAL_SAR_FUSION for ALL grounding, area, and comparative sequences.

Output strictly valid JSON:
{{
  "intent_analysis": {{
    "intent": "vqa | caption | grounding | measurement | change_analysis | multimodal_analysis | comparative_spatial_analysis",
    "target": "extracted natural-language target string or 'scene'",
    "operation": "describe | locate | quantify | compare | rank_density"
  }},
  "task_label": "Official Task Name",
  "tools": [
    {{"tool": "TOOL_NAME", "input": "target parameter"}}
  ]
}}"""
    resp = call_llm_orchestrator(intent_prompt, json_mode=True, api_key=api_key)
    if not resp:
        return None
    try:
        plan = json.loads(resp)
        if "tools" in plan and len(plan["tools"]) > 0:
            return plan
    except Exception:
        return None
    return None

def _run_heuristic_router(request) -> Tuple[dict, List[dict]]:
    """Deterministic NLP router ensuring graceful recovery when LLM APIs are unreachable."""
    q = request.query.lower()
    
    # Target extraction
    target = re.sub(
        r'\b(spot|the|area|where|there|is|more|most|highest|concentration|of|part|which|appears|in|this|image|find|highlight|locate|calculate|coverage|show|me|has|between|two|dates)\b', 
        '', 
        q
    ).strip()
    if not target:
        target = "features"

    first_tool = "OPTICAL_SAR_FUSION" if request.has_sar else "RS_GROUNDING"

    if any(w in q for w in ["most", "more structural", "highest density", "more development", "concentrated"]):
        intent = {
            "intent": "comparative_spatial_analysis",
            "target": target,
            "operation": "rank_density"
        }
        tools = [
            {"tool": first_tool, "input": target},
            {"tool": "SPATIAL_COMPARATOR", "input": target}
        ]
        return intent, tools

    sar_keywords = ["sar", "radar", "microwave", "backscatter", "cross-modal", "fusion", "sentinel-1", "complement"]
    if any(w in q for w in sar_keywords) and request.has_sar:
        return {"intent": "multimodal_analysis", "target": target, "operation": "fusion"}, [{"tool": "OPTICAL_SAR_FUSION", "input": target}]
        
    fusion_keywords = ["both", "together", "combine", "complement"]
    if request.has_sar and any(w in q for w in fusion_keywords):
        return {"intent": "multimodal_analysis", "target": target, "operation": "fusion"}, [{"tool": "OPTICAL_SAR_FUSION", "input": target}]

    change_keywords = ["change", "different", "increased", "decreased", "evolved", "transform", "differ", "modification", "altered", "growth", "expansion", "shrink", "new", "construction", "before", "after", "deforest"]
    if any(w in q for w in change_keywords) and request.has_bitemporal:
        return {"intent": "change_analysis", "target": target, "operation": "compare"}, [{"tool": "CHANGE_DETECTION", "input": target}]

    if any(w in q for w in ["how much", "percentage", "area", "extent", "hectares"]):
        return {"intent": "measurement", "target": target, "operation": "quantify"}, [
            {"tool": first_tool, "input": target},
            {"tool": "AREA_CALCULATOR", "input": "REGIONS"}
        ]

    if any(w in q for w in ["highlight", "mark", "locate", "find", "where are", "spot"]):
        return {"intent": "grounding", "target": target, "operation": "locate"}, [{"tool": first_tool, "input": target}]

    if any(w in q for w in ["describe", "caption", "overview", "summarize"]):
        return {"intent": "caption", "target": "scene", "operation": "describe"}, [{"tool": "RS_CAPTION", "input": request.query}]

    return {"intent": "vqa", "target": target, "operation": "answer"}, [{"tool": "RS_VQA", "input": request.query}]

def route_query(request, settings: Settings) -> Tuple[str, dict, list]:
    api_key = getattr(settings, 'OPENAI_API_KEY', "")
    
    plan = _orchestrate_with_llm(request, api_key)
    if plan and "intent_analysis" in plan and "tools" in plan:
        task_label = plan.get("task_label", "LLM Orchestrated Task")
        return task_label, plan["intent_analysis"], plan["tools"]
        
    intent_info, tools_list = _run_heuristic_router(request)
    return "Heuristic Task", intent_info, tools_list

def synthesize_final_answer(query: str, raw_observations: str, api_key: str) -> Optional[str]:
    """Uses the LLM orchestrator to synthesize a professional final response from raw tool outputs."""
    prompt = (
        "You are SatQuery AI, an expert Remote Sensing Intelligence Assistant. "
        "I will provide you with the user's original query and the raw data observations collected by our specialized backend tools (CV algorithms, Grounding modules, etc.). "
        "Your task is to synthesize these raw observations into a single, cohesive, highly professional, and easy-to-understand response for the user. "
        "Write in a confident, authoritative geospatial intelligence tone (e.g., 'Analysis of the telemetry indicates...', 'Spatial footprints reveal...'). "
        "If the Raw Tool Observations contain an error message (like 429 Too Many Requests, API Error, or missing data), you MUST explicitly state that the system is unable to complete the analysis due to a temporary service disruption or missing telemetry, rather than hallucinating an answer. Do not invent or hallucinate new data! "
        "CRITICAL: Keep your final response strictly concise. It MUST be a single paragraph of exactly 4 to 7 lines long. Do not exceed this length.\n\n"
        f"User Query: {query}\n"
        f"Raw Tool Observations: {raw_observations}\n\n"
        "Provide the final synthesized response:"
    )
    return call_llm_orchestrator(prompt, json_mode=False, api_key=api_key)
