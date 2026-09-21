"""Score completed exports against separately retained synthetic speed truth."""
import argparse
import json
from pathlib import Path

from traffic_ai.speed_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, action="append", required=True)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.pack, args.record, args.runs_root, args.protocol, args.output, args.audit)
    print(json.dumps({"passed": result["passed"], "total": result["total"], "all_passed": result["all_passed"],
                      "cases": [{"scenario": c["scenario"], "state": c["state"], "passed": c["passed"], "metrics": c.get("metrics")} for c in result["cases"]]}))


if __name__ == "__main__":
    main()
