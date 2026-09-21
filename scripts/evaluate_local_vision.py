"""Measure a declared synthetic image set; evaluation labels never enter the model.

This is a small diagnostic sweep, not a traffic-accuracy benchmark. JSON case
labels are provided by the operator after inspection, not inferred by this script.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp.media import load_image
from traffic_ai.bounded_vision import FEATURES, UNCERTAINTIES, review_image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(record, case):
    """Separate runtime completion, unsupported claims, misses and uncertainty."""
    completed = record["state"] == "complete"
    selection = record.get("selection", {})
    selected = set(selection.get("visible_features", [])) if completed else set()
    allowed = set(case["allowed_features"]) | {"unknown"}
    required = set(case.get("required_features", []))
    uncertainty = set(selection.get("uncertainty_codes", [])) if completed else set()
    return {"completed": completed, "unsupported_features": sorted(selected-allowed),
            "missing_required_features": sorted(required-selected) if completed else [],
            "missing_expected_uncertainties": sorted(set(case.get("expected_uncertainties", []))-uncertainty) if completed else [],
            "supported_claims_only": completed and not selected-allowed,
            "feature_target_met": completed and not selected-allowed and required <= selected,
            "uniform_input_abstention": record.get("input_guard") == "uniform_review_image" and not completed and "observation" not in record,
            "fallback_has_no_observation": completed or ("observation" not in record and "selection" not in record),
            "owned_runtime_stopped": record.get("runtime", {}).get("owned_server_stopped") is True}


def load_cases(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("synthetic") is not True or not isinstance(document.get("cases"), list):
        raise ValueError("This diagnostic requires explicitly synthetic cases.")
    cases = document["cases"]
    if not 1 <= len(cases) <= 12:
        raise ValueError("Choose between one and twelve diagnostic cases.")
    ids = set()
    for case in cases:
        identifier = case.get("id", "")
        if not isinstance(identifier, str) or not identifier or not all(c.isascii() and (c.isalnum() or c in "_-") for c in identifier) or identifier in ids:
            raise ValueError("Case IDs must be unique simple names.")
        ids.add(identifier)
        for field, allowed in (("allowed_features", FEATURES), ("required_features", FEATURES), ("expected_uncertainties", UNCERTAINTIES)):
            values = case.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
                raise ValueError("Unsupported diagnostic expectation.")
        if "allowed_features" not in case or not set(case.get("required_features", [])) <= set(case["allowed_features"]):
            raise ValueError("Required features must be within the allowed categories.")
        image = Path(case["image"])
        if not image.is_absolute():
            image = Path(path).resolve().parent/image
        case["image"] = str(image.resolve())
        if sha(image) != case["image_sha256"]:
            raise ValueError("A diagnostic input changed before inference.")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = load_cases(args.manifest)
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    sources = {Path(c["image"]) for c in manifest["cases"]}
    sources.update(Path(c["source"]) for c in manifest["cases"] if c.get("source"))
    source_hashes = {str(p):sha(p) for p in sources}
    record = {"kind":"synthetic_vision_reliability_diagnostic", "created_at":datetime.now(timezone.utc).isoformat(),
              "synthetic":True, "manifest_sha256":sha(args.manifest), "cases":[],
              "source_files":source_hashes, "runtime_source_sha256":sha(ROOT/"traffic_ai/bounded_vision.py"),
              "label_policy":"Allowed/required categories and uncertainty labels are used only after inference. They are never included in the model request.",
              "limitation":"Small selected frames from one invented road and car; no general, real-world, OCR, grounding or traffic-rule accuracy is established."}
    receipt = args.output/"reliability.json"
    for case in manifest["cases"]:
        result = review_image(load_image(case["image"]), case.get("timestamp",0.), output=args.output/case["id"], synthetic=True)
        entry = {"id":case["id"], "evaluation":case, "record":result, "score":score(result,case)}
        record["cases"].append(entry)
        record["source_files_unchanged"] = all(sha(path)==value for path,value in source_hashes.items())
        record["runtime_source_unchanged"] = sha(ROOT/"traffic_ai/bounded_vision.py")==record["runtime_source_sha256"]
        record["summary"] = {key:sum(bool(c["score"][key]) for c in record["cases"]) for key in (
            "completed", "supported_claims_only", "feature_target_met", "uniform_input_abstention", "fallback_has_no_observation", "owned_runtime_stopped")}
        record["summary"]["attempts"] = len(record["cases"])
        receipt.write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(json.dumps({"case":case["id"], "state":result["state"], "seconds":result["total_seconds"],
                          "features":result.get("selection",{}).get("visible_features",[]), "score":entry["score"]}),flush=True)
        if not record["source_files_unchanged"] or not record["runtime_source_unchanged"]:
            break
    print(json.dumps({"summary":record["summary"], "receipt":str(receipt)}))
    # A completed diagnostic can contain model mistakes. Preserve those results
    # instead of presenting a single success/failure accuracy score.
    return int(not record["source_files_unchanged"] or not record["runtime_source_unchanged"] or
               any(not c["score"]["fallback_has_no_observation"] or not c["score"]["owned_runtime_stopped"] for c in record["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
