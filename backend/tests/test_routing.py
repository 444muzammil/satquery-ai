import pytest
from agent.router import _run_heuristic_router
from unittest.mock import MagicMock

class MockRequest:
    def __init__(self, query, has_sar=False, has_bitemporal=False):
        self.query = query
        self.has_sar = has_sar
        self.has_bitemporal = has_bitemporal

def test_heuristic_router_fusion():
    req = MockRequest("Combine the optical and SAR data to find water", has_sar=True)
    intent, tools = _run_heuristic_router(req)
    assert tools[0]["tool"] == "OPTICAL_SAR_FUSION"

def test_heuristic_router_change():
    req = MockRequest("What changed between these images?", has_bitemporal=True)
    intent, tools = _run_heuristic_router(req)
    assert tools[0]["tool"] == "CHANGE_DETECTION"

def test_heuristic_router_grounding():
    req = MockRequest("Locate the urban structures")
    intent, tools = _run_heuristic_router(req)
    assert tools[0]["tool"] == "RS_GROUNDING"

def test_heuristic_router_vqa():
    req = MockRequest("What is the primary land cover?")
    intent, tools = _run_heuristic_router(req)
    assert tools[0]["tool"] == "RS_VQA"
