"""Repeat imported-v2 backend acceptance in a copied primary-app snapshot.

Uses synthetic evidence, authenticated loopback API tests and mocked provider
faults. No live model, preview server, HTML rendering or primary writes occur.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from validate_ai_snapshot import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        assembled = validate(args.primary, args.output, run_tests=False)
        output, source = args.output.resolve(), Path(assembled["snapshot"])
        scratch = output/"runtime-temp"
        scratch.mkdir()
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE":"1", "TEMP":str(scratch), "TMP":str(scratch),
               "TRAFFIC_AI_HOME":str(output/"ai-runtime")}
        command = [sys.executable, "-m", "pytest", "tests/test_review_app_acceptance.py", "tests/test_evidence_bridge.py",
                   "tests/test_review_integration.py", "tests/test_review_grounding.py", "tests/test_bounded_vision.py", "tests/test_local_vision.py",
                   "tests/test_vision_robustness.py", "tests/test_vision_reliability.py",
                   "tests/test_vision_validation_cases.py",
                   "--basetemp", str(output/"pytest-temp"), "--tb=short"]
        with (output/"app-acceptance.log").open("wb") as log:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        primary = args.primary.resolve()
        changed = [name for name, sha in assembled["primary_files"].items() if name.endswith(".py") and (
            not (primary/name).is_file() or hashlib.sha256((primary/name).read_bytes()).hexdigest() != sha)]
        record = {"kind":"imported_v2_review_backend_acceptance", "created_at":datetime.now(timezone.utc).isoformat(),
                  "snapshot":str(source), "primary":str(primary), "command":command,
                  "test_exit_code":result.returncode, "primary_python_sources_unchanged":not changed,
                  "changed_primary_python_paths":changed, "ai_overlay":assembled["ai_overlay"],
                  "passed":result.returncode == 0 and not changed,
                  "classification":"Real loopback HTTP; synthetic evidence; mocked provider outputs. No live model or browser rendering.",
                  "standalone_html_visual_qa":"Still unverified after file: protocol denial; this test does not rehost/render it."}
        (output/"app-acceptance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in record.items() if k not in {"ai_overlay","command"}}))
        return int(not record["passed"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("App acceptance failed: "+str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
