"""Replay saved exports and source faults offline; no model, media decoding or server.

Reviews containing private thumbnails stay in the requested local output. The
aggregate acceptance receipt contains counts and fingerprints, not frame pixels.
Faults use copies of an explicitly synthetic fixture. Originals are read-only.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from traffic_ai.evidence_review import ReviewError,load_review
from traffic_ai.review_cli import export_review,validate_review_bundle
from traffic_ai.review_grounding import pointer_value,validate_grounding


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_sources(run,document):
    return {f["path"]:(run/f["path"]).read_bytes() for f in document["source"]["files"]
            if f["path"] in {"motion-incidents.jsonl","motion-incidents.json","motion-tracks.csv"}}


def evaluate(runs,fixture,output,baseline=None):
    runs=[p.resolve() for p in runs];fixture=fixture.resolve();output=output.resolve()
    if output.exists() or any(output==p or p in output.parents or output in p.parents for p in runs+[fixture]):
        raise ValueError("Choose a new output directory separate from all source runs.")
    fixture_doc=load_review(fixture)
    if not fixture_doc["events"] or not all(e["event"].get("synthetic") is True for e in fixture_doc["events"]):
        raise ValueError("Fault injection requires an explicitly synthetic completed fixture.")
    output.mkdir(parents=True)
    checks=[];records=[];originals={}
    def check(name,passed,**details):
        checks.append({"name":name,"passed":bool(passed),**details})
    for run in runs:
        before=load_review(run)
        for ref in before["source"]["files"]:originals[run/ref["path"]]=ref["sha256"]
        destination=output/("review-"+run.name)
        doc=export_review(run,destination)
        marker=validate_review_bundle(destination)
        sources=verified_sources(run,doc)
        ground=validate_grounding(doc,sources)
        check(run.name+":unchanged_events",[e["event"] for e in before["events"]]==[e["event"] for e in doc["events"]])
        check(run.name+":all_facts_have_exact_citations",all(
            f.get("citations") and all(pointer_value(e if f.get("pointer_base")=="review" else e["event"],c["pointer"])
                ==(c["recorded"],c.get("value")) for c in f["citations"])
            for e in doc["events"] for f in e["facts"]))
        check(run.name+":offline_no_inference",all(e["language_model"]["state"]=="not_requested" for e in doc["events"]))
        check(run.name+":complete_bundle",marker["complete"] is True)
        records.append({"run_id":run.name,"summary":doc["summary"],"grounding":ground,
                        "conditions":dict(Counter(e["event"].get("measurements",{}).get("condition","not_recorded") for e in doc["events"])),
                        "source_files":doc["source"]["files"],"review_files":marker["files"]})
        for key in ("text","citation","event","coverage","explanation","csv"):
            changed=deepcopy(doc)
            if not changed["events"]:continue
            event=changed["events"][0]
            if key=="text":event["facts"][0]["text"]="Invented confirmed violation."
            elif key=="citation":event["facts"][0]["citations"][0]["value"]="invented_status"
            elif key=="event":event["event"]["measurements"]["speed"]=999
            elif key=="coverage":changed["events"].pop()
            elif key=="explanation":event["explanation"].append("Invented event.")
            elif event.get("track_context") and event["track_context"]["samples"]:
                event["track_context"]["samples"][0]["values"]["speed"]="999"
            else:continue
            try:validate_grounding(changed,sources)
            except ReviewError:rejected=True
            else:rejected=False
            check(run.name+":reject_changed_"+key,rejected)
    if baseline:
        previous=json.loads(baseline.read_text(encoding="utf-8"))
        current=json.loads((output/("review-"+runs[0].name)/"review.json").read_text(encoding="utf-8"))
        check("primary_saved_review_events_preserved",[e["event"] for e in previous["events"]]==[e["event"] for e in current["events"]])
    fixture_files={ref["path"]:(fixture/ref["path"]).read_bytes() for ref in fixture_doc["source"]["files"]}
    for name in fixture_files:originals[fixture/name]=hashlib.sha256(fixture_files[name]).hexdigest()
    for fault in ("missing_events","corrupt_stream","missing_status","corrupt_csv","missing_csv","missing_thumbnail"):
        files=dict(fixture_files)
        if fault=="missing_events":files.pop("motion-incidents.jsonl",None);files.pop("motion-incidents.json",None)
        elif fault=="corrupt_stream":files["motion-incidents.jsonl"]=files["motion-incidents.jsonl"].rstrip(b"\n")
        elif fault=="missing_status":files.pop("status.json")
        elif fault=="corrupt_csv":files["motion-tracks.csv"]=b"invalid,columns\n1,2\n"
        elif fault=="missing_csv":files.pop("motion-tracks.csv",None)
        else:
            for name in list(files):
                if name.lower().endswith(".png"):files.pop(name)
        source=output/("fault-"+fault);source.mkdir()
        for name,data in files.items():(source/name).write_bytes(data)
        destination=output/("result-"+fault)
        try:doc=export_review(source,destination)
        except (ReviewError,OSError):
            check(fault+":withheld_no_completed_bundle",fault not in {"missing_csv","missing_thumbnail"} and not (destination/"review-complete.json").exists())
        else:
            missing=doc["grounding"]["missing_optional_sources"]
            correct=("motion-tracks.csv" in missing and all(e.get("stop_review",{}).get("csv_coverage","unavailable")=="unavailable" for e in doc["events"])) if fault=="missing_csv" else any(p.endswith(".png") for p in missing) if fault=="missing_thumbnail" else False
            check(fault+":explicit_unavailable_support",correct and [e["event"] for e in doc["events"]]==[e["event"] for e in fixture_doc["events"]])
    check("all_consumed_original_source_bytes_preserved",all(path.is_file() and digest(path)==sha for path,sha in originals.items()))
    result={"kind":"saved_evidence_grounding_acceptance","created_at":datetime.now(timezone.utc).isoformat(),
            "classification":"Offline real saved-export replay plus synthetic-copy faults; no live inference, accuracy or legal ground truth.",
            "private_artifact_policy":"Local review directories may contain private thumbnail pixels; distribute only this aggregate receipt and source code.",
            "passed":all(c["passed"] for c in checks),"checks":checks,"runs":records,
            "limits":["Source correspondence does not establish detector accuracy, measured road calibration, footage authenticity or a violation.",
                      "Missing optional CSV or thumbnails remain unresolved; required/corrupt evidence withholds review publication.",
                      "Standalone HTML browser rendering remains unverified after file-protocol denial; markup has automated tests."]}
    (output/"acceptance.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,action="append",required=True)
    parser.add_argument("--fault-fixture",type=Path,required=True)
    parser.add_argument("--baseline-review",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:result=evaluate(args.run,args.fault_fixture,args.output,args.baseline_review)
    except (OSError,ValueError) as exc:
        print("Grounding evaluation failed: "+str(exc),file=sys.stderr);return 2
    print(json.dumps({"passed":result["passed"],"checks":len(result["checks"]),"failed":[c["name"] for c in result["checks"] if not c["passed"]],
                      "receipt":str(args.output/"acceptance.json")}))
    return int(not result["passed"])


if __name__=="__main__":raise SystemExit(main())
