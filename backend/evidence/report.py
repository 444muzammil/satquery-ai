"""
SatQuery AI — Report Generator

Produces clean, structured reports for download and audit.
Extracted from main.py and enhanced with downloadable format support.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("satquery.evidence.report")


def generate_report(
    metadata: Dict[str, Any],
    query: str,
    task: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a clean, auditable analysis report.

    Args:
        metadata: Image metadata
        query: User's query
        task: Task that was executed
        result: Analysis result dict

    Returns:
        Structured report dict suitable for JSON download
    """
    cleaned_meta = {str(k).strip().lower(): v for k, v in metadata.items()}

    return {
        "project_id": "SIH26167",
        "project_name": "SatQuery AI",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": query.strip(),
        "task_executed": task,
        "metadata": cleaned_meta,
        "models_used": result.get("model_used", ""),
        "statistics": result.get("evidence", {}).get("stats", {}),
        "confidence_estimate": f"{result.get('confidence', 0.0)}%",
        "model_response": result.get("answer", "").strip(),
        "evidence_type": result.get("evidence", {}).get("type", "text_only"),
        "regions_count": len(result.get("evidence", {}).get("regions", [])),
        "execution_trace": result.get("trace", []),
    }


def generate_downloadable_report(
    metadata: Dict[str, Any],
    query: str,
    task: str,
    result: Dict[str, Any],
) -> str:
    """
    Generate a JSON string for frontend download.

    Returns:
        Pretty-printed JSON string
    """
    report = generate_report(metadata, query, task, result)
    return json.dumps(report, indent=2, default=str)
