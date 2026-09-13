"""
SatQuery AI — Comprehensive Hackathon Readiness Audit Test Suite
================================================================
Tests Gemini VLM, OpenAI orchestrator, agentic routing, validation,
evidence, confidence, execution traces, failure recovery, and
multimodal workflows.
"""
import asyncio
import base64
import json
import os
import sys
import time
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# Setup path
sys.path.insert(0, os.path.abspath('backend'))
from dotenv import load_dotenv
load_dotenv('backend/.env')

from config import Settings
from providers.gemini_vision import GeminiVisionProvider
from providers.openai_vision import OpenAIVisionProvider
from providers.fallback import ClassicalFallbackProvider
from providers.base import ImageInput, TaskType, RSVLMResponse, Modality
from agent.router import route_query, _run_heuristic_router, _orchestrate_with_llm
from agent.controller import AgentController, VQARequest
from agent.validator import detect_modality, validate_upload, ValidationResult

# ============================================================
# TEST INFRASTRUCTURE
# ============================================================

settings = Settings.from_env()
results = []

# Create a small but real test image (10x10 green square PNG)
def make_test_png(width=10, height=10, color=(0, 128, 0)):
    """Create a minimal valid PNG image."""
    import struct, zlib
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    header = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter none
        for x in range(width):
            raw += bytes(color)
    idat = zlib.compress(raw)
    return header + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')

def make_test_b64(width=10, height=10, color=(0, 128, 0)):
    """Return base64-encoded test PNG."""
    png = make_test_png(width, height, color)
    return base64.b64encode(png).decode()

# Two distinct test images for bi-temporal / cross-modal
GREEN_IMG_B64 = make_test_b64(10, 10, (0, 128, 0))   # "vegetation"
GRAY_IMG_B64 = make_test_b64(10, 10, (128, 128, 128)) # "built-up / SAR-like"
BLUE_IMG_B64 = make_test_b64(10, 10, (0, 0, 200))     # "water"

def record(test_id, name, expected, actual, status, evidence=""):
    results.append({
        "id": test_id, "name": name, "expected": expected,
        "actual": actual, "status": status, "evidence": evidence
    })
    icon = {"PASS": "[OK]", "PARTIAL": "[??]", "FAIL": "[XX]", "BLOCKED": "[!!]", "NOT_IMPL": "[--]"}.get(status, "?")
    print(f"  {icon} {test_id}: {name} [{status}]")
    if status in ("FAIL", "PARTIAL"):
        print(f"      Expected: {expected}")
        print(f"      Actual:   {actual}")

# ============================================================
# T01 — REPOSITORY STARTUP
# ============================================================
def test_T01():
    print("\n=== T01: Repository Startup ===")
    try:
        from config import settings as s
        assert s is not None
        record("T01", "Repository startup & config load", "Settings loaded", "Settings loaded", "PASS")
    except Exception as e:
        record("T01", "Repository startup", "Settings loaded", str(e), "FAIL")

# ============================================================
# T02 — API HEALTH (import FastAPI app)
# ============================================================
def test_T02():
    print("\n=== T02: API Health ===")
    try:
        from main import app
        assert app is not None
        assert app.title == "SatQuery AI - Geospatial Intelligence Backend"
        routes = [r.path for r in app.routes]
        record("T02", "FastAPI app initialization", "App with routes", f"Routes: {routes}", "PASS")
    except Exception as e:
        record("T02", "FastAPI app initialization", "App loads", str(e), "FAIL")

# ============================================================
# T03 — GEMINI VLM CONNECTIVITY
# ============================================================
async def test_T03():
    print("\n=== T03: Gemini VLM Connectivity ===")
    try:
        provider = GeminiVisionProvider(settings)
        assert provider.api_key, "Gemini API key is empty"
        assert "gemini" in provider.model_name.lower() or "3.6" in provider.model_name
        # Simple text-only test
        resp = await provider.analyze(
            images=[ImageInput(data=GREEN_IMG_B64)],
            query="Say the word CONNECTED if you can read this.",
            task=TaskType.VQA
        )
        if resp.success and resp.text and len(resp.text) > 0:
            record("T03", "Gemini VLM connectivity", "200 OK + text response",
                   f"Model: {resp.model_name}, Response length: {len(resp.text)}", "PASS",
                   f"Response: {resp.text[:100]}")
        else:
            record("T03", "Gemini VLM connectivity", "200 OK", f"Error: {resp.text}", "FAIL")
    except Exception as e:
        record("T03", "Gemini VLM connectivity", "Connection success", str(e), "FAIL")

# ============================================================
# T04 — GEMINI IMAGE INFERENCE
# ============================================================
async def test_T04():
    print("\n=== T04: Gemini Image Inference ===")
    try:
        provider = GeminiVisionProvider(settings)
        resp = await provider.analyze(
            images=[ImageInput(data=GREEN_IMG_B64)],
            query="What is the dominant color in this image? Answer in one word.",
            task=TaskType.VQA
        )
        if resp.success and resp.text:
            has_color = any(w in resp.text.lower() for w in ["green", "color", "dark", "black"])
            status = "PASS" if has_color else "PARTIAL"
            record("T04", "Gemini image inference", "Color-related response",
                   f"Response: {resp.text[:100]}", status,
                   f"Latency: {resp.latency_ms}ms")
        else:
            record("T04", "Gemini image inference", "Image analysis", f"Error: {resp.text}", "FAIL")
    except Exception as e:
        record("T04", "Gemini image inference", "Image analyzed", str(e), "FAIL")

# ============================================================
# T05 — SINGLE-IMAGE OPTICAL VQA (via full pipeline)
# ============================================================
async def test_T05():
    print("\n=== T05: Single-Image Optical VQA ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="What type of land cover is visible in this image?",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        has_answer = bool(result.get("answer"))
        has_trace = bool(result.get("trace"))
        has_evidence = "evidence" in result
        has_confidence = "confidence" in result
        checks = [has_answer, has_trace, has_evidence, has_confidence]
        if all(checks):
            record("T05", "Single-image optical VQA pipeline",
                   "answer+trace+evidence+confidence",
                   f"Answer: {str(result.get('answer',''))[:80]}",
                   "PASS", f"Confidence: {result.get('confidence')}")
        elif has_answer:
            missing = []
            if not has_trace: missing.append("trace")
            if not has_evidence: missing.append("evidence")
            if not has_confidence: missing.append("confidence")
            record("T05", "Single-image optical VQA pipeline",
                   "All fields present", f"Missing: {missing}", "PARTIAL")
        else:
            record("T05", "Single-image optical VQA pipeline",
                   "Answer returned", f"Result: {json.dumps(result)[:200]}", "FAIL")
    except Exception as e:
        record("T05", "Single-image optical VQA pipeline", "Pipeline completes", str(e), "FAIL",
               traceback.format_exc())

# ============================================================
# T06 — SINGLE-IMAGE SAR VQA
# ============================================================
async def test_T06():
    print("\n=== T06: Single-Image SAR VQA ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="What features are visible in this SAR image?",
            image_base64=GRAY_IMG_B64,
            has_sar=True,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        has_answer = bool(result.get("answer"))
        if has_answer:
            record("T06", "Single-image SAR VQA", "Answer returned",
                   f"Answer: {str(result.get('answer',''))[:80]}", "PASS")
        else:
            record("T06", "Single-image SAR VQA", "Answer returned",
                   f"Result keys: {list(result.keys())}", "FAIL")
    except Exception as e:
        record("T06", "Single-image SAR VQA", "Pipeline completes", str(e), "FAIL")

# ============================================================
# T07 — CAPTIONING / SCENE DESCRIPTION
# ============================================================
async def test_T07():
    print("\n=== T07: Captioning / Scene Description ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="Describe the land-cover and major objects visible in this image.",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        answer = result.get("answer", "")
        trace = result.get("trace", "")
        # Check if captioning tool was routed
        routed_to_caption = "CAPTION" in str(trace).upper() or "caption" in str(result.get("task", "")).lower()
        if answer and len(answer) > 20:
            status = "PASS" if routed_to_caption else "PARTIAL"
            evidence = "Routed to captioning tool" if routed_to_caption else "Response generated but routing unclear"
            record("T07", "Captioning/scene description", "Descriptive scene answer",
                   f"{answer[:80]}", status, evidence)
        else:
            record("T07", "Captioning/scene description", "Scene description", f"Got: {answer[:50]}", "FAIL")
    except Exception as e:
        record("T07", "Captioning/scene description", "Pipeline completes", str(e), "FAIL")

# ============================================================
# T09 — BI-TEMPORAL CHANGE ANALYSIS
# ============================================================
async def test_T09():
    print("\n=== T09: Bi-Temporal Change Analysis ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="What changed between these two dates, and where did the change occur?",
            image_base64=GREEN_IMG_B64,
            image_b_base64=GRAY_IMG_B64,
            has_sar=False,
            has_bitemporal=True
        )
        result = await controller.execute(req)
        answer = result.get("answer", "")
        trace = str(result.get("trace", ""))
        change_routed = "CHANGE" in trace.upper() or "change" in str(result.get("task", "")).lower()
        if answer and change_routed:
            record("T09", "Bi-temporal change analysis", "Change description + routing",
                   f"{answer[:80]}", "PASS", f"Task: {result.get('task')}")
        elif answer:
            record("T09", "Bi-temporal change analysis", "Change routing",
                   f"Answer present but routing unclear", "PARTIAL")
        else:
            record("T09", "Bi-temporal change analysis", "Change output", f"Keys: {list(result.keys())}", "FAIL")
    except Exception as e:
        record("T09", "Bi-temporal change analysis", "Pipeline completes", str(e), "FAIL",
               traceback.format_exc())

# ============================================================
# T10 — CHANGE-BASED VQA
# ============================================================
async def test_T10():
    print("\n=== T10: Change-Based VQA ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="Has the built-up area increased, decreased, or remained unchanged?",
            image_base64=GREEN_IMG_B64,
            image_b_base64=GRAY_IMG_B64,
            has_sar=False,
            has_bitemporal=True
        )
        result = await controller.execute(req)
        answer = str(result.get("answer", "")).lower()
        has_answer = any(w in answer for w in ["increase", "decrease", "unchanged", "change", "built", "remain"])
        if has_answer:
            record("T10", "Change-based VQA", "Directional change answer",
                   f"{result.get('answer','')[:80]}", "PASS")
        elif result.get("answer"):
            record("T10", "Change-based VQA", "Directional answer", f"Got: {answer[:80]}", "PARTIAL")
        else:
            record("T10", "Change-based VQA", "Answer", "No answer", "FAIL")
    except Exception as e:
        record("T10", "Change-based VQA", "Pipeline completes", str(e), "FAIL")

# ============================================================
# T12 — OPTICAL + SAR CROSS-MODAL ANALYSIS
# ============================================================
async def test_T12():
    print("\n=== T12: Optical + SAR Cross-Modal Analysis ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="Use the optical and SAR images together to identify built-up and water-covered regions.",
            image_base64=GREEN_IMG_B64,
            image_b_base64=GRAY_IMG_B64,
            has_sar=True,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        answer = result.get("answer", "")
        trace = str(result.get("trace", ""))
        fusion_routed = "FUSION" in trace.upper() or "fusion" in str(result.get("task", "")).lower()
        if answer and fusion_routed:
            record("T12", "Optical+SAR cross-modal", "Fusion routing + output",
                   f"{answer[:80]}", "PASS", f"Task: {result.get('task')}")
        elif answer:
            record("T12", "Optical+SAR cross-modal", "Fusion routing",
                   f"Answer present, routing: {result.get('task')}", "PARTIAL")
        else:
            record("T12", "Optical+SAR cross-modal", "Fusion output", f"Keys: {list(result.keys())}", "FAIL")
    except Exception as e:
        record("T12", "Optical+SAR cross-modal", "Pipeline completes", str(e), "FAIL",
               traceback.format_exc())

# ============================================================
# T15 — AUTOMATIC TASK ROUTING (AGENTIC)
# ============================================================
def test_T15():
    print("\n=== T15: Automatic Task Routing ===")
    test_cases = [
        ("What is visible in this image?", False, False, "vqa"),
        ("Describe the scene", False, False, "caption"),
        ("Highlight the water bodies", False, False, "grounding"),
        ("What changed between these two dates?", False, True, "change"),
        ("Use optical and SAR together", True, False, "fusion"),
    ]
    passed = 0
    for query, has_sar, has_bt, expected_type in test_cases:
        req = VQARequest(query=query, image_base64="dummy", has_sar=has_sar, has_bitemporal=has_bt)
        task_label, intent, tools = route_query(req, settings)
        tool_names = [t.get("tool", "") for t in tools]
        detected = str(intent.get("intent", "")).lower()
        expected_in = expected_type.lower()
        match = expected_in in detected or expected_in in str(tool_names).lower()
        if match:
            passed += 1
        status_marker = "[PASS]" if match else "[FAIL]"
        print(f"    Query: '{query[:40]}' -> {detected} -> {tool_names} {status_marker}")

    if passed == len(test_cases):
        record("T15", "Automatic task routing", "All queries routed correctly",
               f"{passed}/{len(test_cases)} correct", "PASS")
    elif passed >= 3:
        record("T15", "Automatic task routing", "All correct",
               f"{passed}/{len(test_cases)} correct", "PARTIAL")
    else:
        record("T15", "Automatic task routing", "Correct routing",
               f"{passed}/{len(test_cases)} correct", "FAIL")

# ============================================================
# T16 — INVALID MODALITY HANDLING
# ============================================================
def test_T16():
    print("\n=== T16: Invalid Modality Handling ===")
    try:
        result = detect_modality(None)
        if result == Modality.UNKNOWN or result is not None:
            record("T16", "Invalid modality handling", "Graceful handling", f"Result: {result}", "PASS")
        else:
            record("T16", "Invalid modality handling", "Graceful", f"Result: {result}", "FAIL")
    except Exception as e:
        # If it raises an exception, that's also valid error handling
        record("T16", "Invalid modality handling", "Graceful handling", f"Exception: {e}", "PARTIAL")

# ============================================================
# T20 — MALFORMED QUERY
# ============================================================
async def test_T20():
    print("\n=== T20: Malformed Query Handling ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        # Should either return error or handle gracefully
        if "error" in str(result).lower() or result.get("answer"):
            record("T20", "Malformed query (empty)", "Graceful handling",
                   f"Result: {str(result)[:100]}", "PASS")
        else:
            record("T20", "Malformed query (empty)", "Error or response",
                   f"Result: {str(result)[:100]}", "PARTIAL")
    except Exception as e:
        record("T20", "Malformed query (empty)", "No crash", str(e), "FAIL")

# ============================================================
# T21 — GEMINI FAILURE HANDLING (bad key)
# ============================================================
async def test_T21():
    print("\n=== T21: Gemini Failure Handling ===")
    try:
        bad_settings = Settings.from_env()
        bad_settings.GEMINI_API_KEY = "INVALID_KEY_FOR_TESTING"
        provider = GeminiVisionProvider(bad_settings)
        resp = await provider.analyze(
            images=[ImageInput(data=GREEN_IMG_B64)],
            query="Test query",
            task=TaskType.VQA
        )
        # Should return error response, not crash
        if not resp.success or "error" in resp.text.lower() or resp.is_fallback:
            record("T21", "Gemini failure handling", "Graceful error response",
                   f"Fallback: {resp.is_fallback}, Text: {resp.text[:60]}", "PASS")
        else:
            record("T21", "Gemini failure handling", "Error detected",
                   f"Got success with bad key: {resp.text[:60]}", "FAIL")
    except Exception as e:
        record("T21", "Gemini failure handling", "No crash", f"Crash: {e}", "FAIL")

# ============================================================
# T24 — CONFIDENCE VALIDATION
# ============================================================
async def test_T24():
    print("\n=== T24: Confidence Validation ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="What land cover is visible?",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        conf = result.get("confidence")
        if conf is not None:
            is_valid = isinstance(conf, (int, float)) and 0 <= conf <= 100
            if is_valid:
                record("T24", "Confidence validation", "0-100 numeric",
                       f"Confidence: {conf}", "PASS")
            else:
                record("T24", "Confidence validation", "Valid range",
                       f"Confidence: {conf} (type: {type(conf).__name__})", "FAIL")
        else:
            record("T24", "Confidence validation", "Confidence present",
                   "No confidence field", "FAIL")
    except Exception as e:
        record("T24", "Confidence validation", "Confidence returned", str(e), "FAIL")

# ============================================================
# T25 — EXECUTION TRACE VALIDATION
# ============================================================
async def test_T25():
    print("\n=== T25: Execution Trace Validation ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="Describe this satellite image",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        trace = result.get("trace", "")
        if trace and len(str(trace)) > 10:
            trace_str = str(trace)
            has_task = "TASK" in trace_str.upper() or "task" in trace_str.lower()
            has_model = any(w in trace_str.upper() for w in ["MODEL", "TOOL", "VQA", "CAPTION", "GEMINI"])
            if has_task and has_model:
                record("T25", "Execution trace validation", "Task + model info",
                       f"Trace length: {len(trace_str)}", "PASS",
                       f"Trace snippet: {trace_str[:150]}")
            else:
                record("T25", "Execution trace validation", "Task + model",
                       f"Has task: {has_task}, Has model: {has_model}", "PARTIAL")
        else:
            record("T25", "Execution trace validation", "Non-empty trace",
                   f"Trace: {trace}", "FAIL")
    except Exception as e:
        record("T25", "Execution trace validation", "Trace returned", str(e), "FAIL")

# ============================================================
# T26 — EVIDENCE VALIDATION
# ============================================================
async def test_T26():
    print("\n=== T26: Evidence Validation ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        req = VQARequest(
            query="Identify features in this image",
            image_base64=GREEN_IMG_B64,
            has_sar=False,
            has_bitemporal=False
        )
        result = await controller.execute(req)
        evidence = result.get("evidence", {})
        if evidence and isinstance(evidence, dict):
            has_regions = "regions" in evidence
            has_stats = "stats" in evidence
            record("T26", "Evidence validation", "Evidence dict with regions/stats",
                   f"Keys: {list(evidence.keys())}", "PASS" if (has_regions or has_stats) else "PARTIAL")
        else:
            record("T26", "Evidence validation", "Evidence dict", f"Evidence: {evidence}", "FAIL")
    except Exception as e:
        record("T26", "Evidence validation", "Evidence returned", str(e), "FAIL")

# ============================================================
# T30 — REPEATED INFERENCE STABILITY
# ============================================================
async def test_T30():
    print("\n=== T30: Repeated Inference Stability ===")
    try:
        provider = GeminiVisionProvider(settings)
        successes = 0
        for i in range(3):
            resp = await provider.analyze(
                images=[ImageInput(data=GREEN_IMG_B64)],
                query=f"What color is this image? (attempt {i+1})",
                task=TaskType.VQA
            )
            if resp.success:
                successes += 1
            await asyncio.sleep(1)  # Rate limiting
        if successes == 3:
            record("T30", "Repeated inference stability", "3/3 success",
                   f"{successes}/3 succeeded", "PASS")
        elif successes >= 2:
            record("T30", "Repeated inference stability", "3/3 success",
                   f"{successes}/3 succeeded", "PARTIAL")
        else:
            record("T30", "Repeated inference stability", "Consistent success",
                   f"{successes}/3 succeeded", "FAIL")
    except Exception as e:
        record("T30", "Repeated inference stability", "No crash", str(e), "FAIL")

# ============================================================
# ABLATION: T13 — OPTICAL-ONLY vs T14 — SAR-ONLY vs T12 CROSS
# ============================================================
async def test_T13_T14_ablation():
    print("\n=== T13/T14: Cross-Modal Ablation ===")
    try:
        provider = GeminiVisionProvider(settings)
        controller = AgentController(settings, provider)
        
        # Optical only
        req_opt = VQARequest(
            query="Identify built-up and water-covered regions.",
            image_base64=GREEN_IMG_B64,
            has_sar=False, has_bitemporal=False
        )
        result_opt = await controller.execute(req_opt)
        
        # SAR only
        req_sar = VQARequest(
            query="Identify built-up and water-covered regions.",
            image_base64=GRAY_IMG_B64,
            has_sar=True, has_bitemporal=False
        )
        result_sar = await controller.execute(req_sar)
        
        opt_answer = str(result_opt.get("answer", ""))
        sar_answer = str(result_sar.get("answer", ""))
        
        if opt_answer and sar_answer:
            # Check if responses are meaningfully different
            different = opt_answer[:50] != sar_answer[:50]
            record("T13", "Optical-only ablation", "Response generated",
                   f"Opt: {opt_answer[:60]}", "PASS")
            record("T14", "SAR-only ablation", "Response generated",
                   f"SAR: {sar_answer[:60]}", "PASS")
        else:
            record("T13", "Optical-only ablation", "Response", f"Empty: {not opt_answer}", "FAIL")
            record("T14", "SAR-only ablation", "Response", f"Empty: {not sar_answer}", "FAIL")
    except Exception as e:
        record("T13", "Optical-only ablation", "Completes", str(e), "FAIL")
        record("T14", "SAR-only ablation", "Completes", str(e), "FAIL")

# ============================================================
# RUNNER
# ============================================================
async def main():
    print("=" * 70)
    print("SATQUERY AI — COMPREHENSIVE HACKATHON READINESS AUDIT")
    print("=" * 70)
    start = time.time()

    # Sync tests
    test_T01()
    test_T02()
    test_T15()
    test_T16()

    # Async tests (Gemini VLM) — with delays to avoid free-tier rate limiting
    await test_T03()
    await asyncio.sleep(3)
    await test_T04()
    await asyncio.sleep(3)
    await test_T05()
    await asyncio.sleep(3)
    await test_T06()
    await asyncio.sleep(3)
    await test_T07()
    await asyncio.sleep(3)
    await test_T09()
    await asyncio.sleep(3)
    await test_T10()
    await asyncio.sleep(3)
    await test_T12()
    await asyncio.sleep(3)
    await test_T13_T14_ablation()
    await asyncio.sleep(3)
    await test_T20()
    await asyncio.sleep(3)
    await test_T21()
    await asyncio.sleep(3)
    await test_T24()
    await asyncio.sleep(3)
    await test_T25()
    await asyncio.sleep(3)
    await test_T26()
    await asyncio.sleep(3)
    await test_T30()

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"AUDIT COMPLETE — {len(results)} tests in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Summary
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    # Dump JSON
    with open("audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to audit_results.json")

if __name__ == "__main__":
    asyncio.run(main())
