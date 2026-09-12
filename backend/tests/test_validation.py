import pytest
from agent.validator import validate_query, validate_task_inputs, detect_modality
from providers.base import Modality

def test_validate_query():
    assert validate_query("What is this?").valid is True
    assert validate_query("").valid is False
    assert validate_query("   ").valid is False
    assert validate_query("A" * 2001).valid is False

def test_validate_task_inputs():
    # Change detection
    assert validate_task_inputs("change_detection", True, True, False).valid is True
    assert validate_task_inputs("change_detection", True, False, False).valid is False
    
    # Fusion
    assert validate_task_inputs("fusion", True, False, True).valid is True
    assert validate_task_inputs("fusion", True, True, False).valid is False
    
    # Standard single image
    assert validate_task_inputs("vqa", True, False, False).valid is True
    assert validate_task_inputs("vqa", False, False, False).valid is False

def test_detect_modality():
    # SAR
    sar_meta = {"bands": 1, "dtype": "float32", "p2": 1.0, "p98": 150.0}
    assert detect_modality(sar_meta) == Modality.SAR
    
    # Optical RGB
    rgb_meta = {"bands": 3, "dtype": "uint8"}
    assert detect_modality(rgb_meta) == Modality.OPTICAL_RGB
    
    # Multispectral
    ms_meta = {"bands": 12, "dtype": "uint16"}
    assert detect_modality(ms_meta) == Modality.MULTISPECTRAL
