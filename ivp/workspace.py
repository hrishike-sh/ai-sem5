"""Small, deterministic helpers for the interactive workspace."""

import json
from pathlib import Path

def comparison_wipe(original, processed, percent):
    """Left is original, right is processed, in the same pixel coordinates."""
    if original.shape != processed.shape:
        raise ValueError("Comparison frames must have matching dimensions.")
    split = round(original.shape[1] * max(0, min(100, percent)) / 100)
    result = processed.copy()
    result[:, :split] = original[:, :split]
    if 0 < split < original.shape[1]:
        result[:, split:split + 1] = 255
    return result


def run_details(run):
    """Read provenance without making a damaged historical report break the UI."""
    run = Path(run)
    for filename in ("report.json", "request.json"):
        try:
            value = json.loads((run / filename).read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, ValueError):
            pass
    return {}


def source_name(details):
    return str(details.get("source", "Unknown source")).replace("\\", "/").rsplit("/", 1)[-1]
