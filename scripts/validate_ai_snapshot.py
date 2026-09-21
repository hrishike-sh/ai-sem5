"""Test the current primary source plus isolated AI additions in a new snapshot.

Only git-listed tracked files and untracked Python source are read from primary.
Ignored models, uploads, generated runs, credentials and its Git metadata stay put.
No source checkout is edited and no preview server is launched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
AI_FILES = ["traffic_ai/evidence_review.py", "traffic_ai/review_cli.py", "traffic_ai/review_llm.py",
            "traffic_ai/review_tracks.py", "traffic_ai/review_stop.py", "tests/test_evidence_review.py",
            "tests/test_review_integration.py", "scripts/validate_ai_review.py", "scripts/validate_ai_snapshot.py",
            "docs/AI_EVIDENCE_REVIEW.md", "traffic_ai/bounded_vision.py", "traffic_ai/local_vision.py",
            "tests/test_local_vision.py", "tests/test_bounded_vision.py", "tests/test_review_app_acceptance.py",
            "scripts/review_local_vision.py", "scripts/validate_review_app.py", "scripts/validate_local_vision.py",
            "docs/LOCAL_VISION_REVIEW.md", "scripts/evaluate_local_vision.py", "tests/test_vision_robustness.py",
            "tests/test_vision_reliability.py", "docs/VISION_RELIABILITY_20261018.md",
            "tests/test_vision_validation_cases.py", "traffic_ai/review_grounding.py", "tests/test_review_grounding.py"]


def _listed(primary, *args):
    value = subprocess.check_output(["git", "ls-files", "-z", *args], cwd=primary)
    return {p.decode("utf-8") for p in value.split(b"\0") if p}


def validate(primary, output, *, run_tests=True):
    primary, output = Path(primary).resolve(), Path(output).resolve()
    if (output.exists() or output == primary or primary in output.parents or output in primary.parents or
            output == ROOT or output in ROOT.parents):
        raise ValueError("Snapshot must be a new directory outside primary and separate from this source root.")
    tracked = _listed(primary)
    untracked = {p for p in _listed(primary, "--others", "--exclude-standard") if
                 p.endswith(".py") and Path(p).parts[0] in {"ivp", "traffic_ai", "tests", "scripts"}}
    # Source only: tracked credential-looking files are also excluded defensively.
    paths = sorted(p for p in tracked | untracked if not any(part.startswith(".env") or part in
                   {".git", ".aws", ".codex", ".agents"} for part in Path(p).parts)
                   and Path(p).suffix.lower() not in {".key", ".pem", ".pt", ".pth", ".safetensors", ".onnx"})
    source, overlay = {}, {}
    output.mkdir(parents=True)
    snapshot = output/"source"
    snapshot.mkdir()
    for name in paths:
        path = primary/name
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(primary):
            raise ValueError("Primary source path escapes its checkout: "+name)
        data = path.read_bytes()
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        source[name] = hashlib.sha256(data).hexdigest()
    for name in AI_FILES:
        data = (ROOT/name).read_bytes()
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        overlay[name] = hashlib.sha256(data).hexdigest()
    unchanged = all((primary/name).is_file() and hashlib.sha256((primary/name).read_bytes()).hexdigest() == sha
                    for name, sha in source.items())
    if not unchanged:
        raise ValueError("Primary source changed while assembling the snapshot; retry in a new output directory.")
    result = {"kind": "isolated_ai_primary_compatibility", "created_at": datetime.now(timezone.utc).isoformat(),
              "primary": str(primary), "snapshot": str(snapshot), "primary_files": source, "ai_overlay": overlay,
              "primary_unchanged_after_copy": unchanged, "test_commands": [],
              "note": "This is a copied local compatibility checkout, not a commit, deployment or detector-accuracy benchmark."}
    if run_tests:
        scratch = output/"runtime-temp"
        scratch.mkdir()
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TEMP": str(scratch), "TMP": str(scratch),
               "TRAFFIC_AI_HOME": str(output/"ai-runtime")}
        commands = [([sys.executable, "-m", "pytest", "--basetemp", str(output/"pytest-temp"), "--tb=short"], "python.log"),
                    (["node", "--test", "tests/web_job.test.cjs"], "javascript.log")]
        for command, logfile in commands:
            with (output/logfile).open("wb") as log:
                completed = subprocess.run(command, cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            result["test_commands"].append({"command": command, "exit_code": completed.returncode, "log": logfile})
    result["primary_unchanged_after_tests"] = all((primary/name).is_file() and
        hashlib.sha256((primary/name).read_bytes()).hexdigest() == sha for name, sha in source.items())
    result["passed"] = (result["primary_unchanged_after_tests"] and
                        all(c["exit_code"] == 0 for c in result["test_commands"]))
    (output/"compatibility.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.primary, args.output, run_tests=not args.assemble_only)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Snapshot validation failed: "+str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"], "snapshot": result["snapshot"], "tests": result["test_commands"],
                      "receipt": str(args.output/"compatibility.json")}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
