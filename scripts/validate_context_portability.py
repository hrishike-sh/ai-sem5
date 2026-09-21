"""Independent saved CSV/order and detached-report audit, with no model or browser.

Only aggregate findings/fingerprints are written. Original runs and reports stay
read-only. Source access is artificially denied during the portability check;
this is static artifact acceptance, not cross-machine browser rendering.
"""
import argparse
import base64
import csv
from datetime import datetime,timezone
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from traffic_ai.review_cli import validate_review_bundle


class Resources(HTMLParser):
    def __init__(self):
        super().__init__();self.resources=[];self.forbidden=[];self.articles=[];self.current=None
    def handle_starttag(self,tag,attrs):
        if tag in {"script","iframe","link","object","embed","base","form"}:self.forbidden.append(tag)
        if tag=="article":self.current=[];self.articles.append(self.current)
        for key,value in attrs:
            if key in {"src","href","action","poster","srcset","data"}:
                self.resources.append(value)
                if self.current is not None and tag=="img":self.current.append(value)
    def handle_endtag(self,tag):
        if tag=="article":self.current=None


def fingerprint(data):return hashlib.sha256(data).hexdigest()


def audit(run,bundle):
    run,bundle=run.resolve(),bundle.resolve()
    marker=validate_review_bundle(bundle)
    doc=json.loads((bundle/"review.json").read_text(encoding="utf-8"))
    raw=(run/"motion-incidents.jsonl").read_bytes()
    lines=raw.splitlines();events=[json.loads(line) for line in lines]
    csv_data=(run/"motion-tracks.csv").read_bytes()
    reader=csv.DictReader(io.StringIO(csv_data.decode("utf-8-sig"),newline=""))
    rows=[{"record_number":n,"line_end":reader.line_num,"values":dict(row)} for n,row in enumerate(reader,1)]
    checks=[];mismatches=[]
    def check(name,passed):checks.append({"name":name,"passed":bool(passed)})
    check("every_event_in_exact_source_order",[r["event"] for r in doc["events"]]==events)
    check("source_fingerprints_match",all((run/f["path"]).is_file() and (run/f["path"]).stat().st_size==f["bytes"] and
          fingerprint((run/f["path"]).read_bytes())==f["sha256"] for f in doc["source"]["files"]))
    references=unresolved=omitted=0
    for index,item in enumerate(doc["events"]):
        if index>=len(events):break
        expected_ref={"file":"motion-incidents.jsonl","line":index+1,"sha256":fingerprint(lines[index])}
        if item["evidence_ref"]!=expected_ref:mismatches.append({"record":index+1,"field":"event_source_reference"})
        event=events[index];identity=event.get("track_id")
        key=str(identity) if type(identity) is int else identity if isinstance(identity,str) and identity.strip() else None
        for field in ("track_context","stop_context"):
            context=item.get(field)
            if context is None:continue
            if field=="track_context":
                start=max(0.,event["timestamp_seconds"]-2.)
                frame=event.get("evidence_timestamp_seconds")
                if type(frame) not in (int,float) or not math.isfinite(frame):frame=event["timestamp_seconds"]
                end=max(event["timestamp_seconds"],frame)+1.
            else:
                proof=event["measurements"]["stop_evidence"];start,end=proof["start_seconds"],proof["end_seconds"]
            # Direct scan independent of the production active-window grouping.
            matching=[row for row in rows if key is not None and row["values"]["track_id"]==key and
                      start-1e-6<=float(row["values"]["timestamp_seconds"])<=end+1e-6]
            shown=matching if len(matching)<=120 else matching[:60]+matching[-60:]
            uncertain=sum(r["values"]["tracking_state"]!="observed" for r in matching)
            duplicates=sum(abs(float(a["values"]["timestamp_seconds"])-float(b["values"]["timestamp_seconds"]))<1e-9
                           for a,b in zip(matching,matching[1:]))
            wanted={"requested_window_seconds":[start,end],"samples":shown,"sample_count":len(matching),"shown_samples":len(shown),
                    "omitted_samples":len(matching)-len(shown),"observed_rows":len(matching)-uncertain,"unresolved_rows":uncertain,
                    "duplicate_timestamps":duplicates,"first_timestamp_seconds":float(matching[0]["values"]["timestamp_seconds"]) if matching else None,
                    "last_timestamp_seconds":float(matching[-1]["values"]["timestamp_seconds"]) if matching else None}
            for name,value in wanted.items():
                if context.get(name)!=value:mismatches.append({"record":index+1,"field":field+"/"+name})
            if context["source"]["sha256"]!=fingerprint(csv_data):mismatches.append({"record":index+1,"field":field+"/source_hash"})
            if uncertain and "unresolved_track_context" not in {w["code"] for w in item["warnings"]}:
                mismatches.append({"record":index+1,"field":field+"/ambiguity_warning"})
            references+=len(matching);unresolved+=uncertain;omitted+=len(matching)-len(shown)
    check("independent_csv_windows_counts_raw_rows_and_event_citations",not mismatches)
    before={f["path"]:f["sha256"] for f in doc["source"]["files"]}
    original_open=Path.open;blocked=[]
    def deny_source(path,*args,**kwargs):
        target=path.resolve()
        if target==run or run in target.parents:
            blocked.append(str(target));raise OSError("Source intentionally unavailable during portable acceptance.")
        return original_open(path,*args,**kwargs)
    # No original files are renamed, deleted, made inaccessible or copied.
    with patch.object(Path,"open",deny_source):
        detached_marker=validate_review_bundle(bundle)
        detached=json.loads((bundle/"review.json").read_text(encoding="utf-8"))
        page=(bundle/"review.html").read_text(encoding="utf-8")
        parser=Resources();parser.feed(page)
        check("source_unavailable_bundle_still_valid",detached_marker==marker and detached==doc and not blocked)
        check("portable_markup_has_no_external_resources",not parser.forbidden and all(
            value.startswith("data:image/png;base64,") for value in parser.resources))
        embedded=True;count=0
        if len(parser.articles)!=len(detached["events"]):embedded=False
        for item,images in zip(detached["events"],parser.articles):
            ref=item.get("thumbnail_ref")
            if ref:
                count+=1
                embedded &= len(images)==1 and fingerprint(base64.b64decode(images[0].split(",",1)[1],validate=True))==ref["sha256"]
            elif images:embedded=False
        check("embedded_thumbnails_match_recorded_fingerprints",embedded)
        check("every_original_event_and_csv_excerpt_embedded",all(
            html.escape(json.dumps(item["event"],ensure_ascii=False,indent=2,allow_nan=False),quote=True) in page and all(
                html.escape(json.dumps(item[field],ensure_ascii=False,indent=2,allow_nan=False),quote=True) in page
                for field in ("track_context","stop_context") if field in item) for item in detached["events"]))
    check("original_consumed_files_unchanged",all(fingerprint((run/name).read_bytes())==sha for name,sha in before.items()))
    return {"run_id":run.name,"passed":all(c["passed"] for c in checks),"checks":checks,"mismatches":mismatches,
            "events":len(events),"status_counts":doc["summary"]["statuses"],"event_timestamp_reversals":sum(
                b["timestamp_seconds"]<a["timestamp_seconds"] for a,b in zip(events,events[1:])),
            "csv_records":len(rows),"csv_tracks":len({r["values"]["track_id"] for r in rows}),
            "unresolved_csv_records":sum(r["values"]["tracking_state"]!="observed" for r in rows),
            "context_row_references":references,"unresolved_context_row_references":unresolved,"omitted_context_row_references":omitted,
            "embedded_thumbnail_references":count,"report_files":marker["files"],
            "source_sha256":{"motion-incidents.jsonl":fingerprint(raw),"motion-tracks.csv":fingerprint(csv_data)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",nargs=2,type=Path,action="append",metavar=("RUN","REVIEW_BUNDLE"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        output=args.output.resolve()
        if output.exists():raise ValueError("Choose a new receipt filename.")
        if any(root.resolve() in output.parents for case in args.case for root in case):
            raise ValueError("Receipt must stay outside the original run and report directories.")
        cases=[audit(run,bundle) for run,bundle in args.case]
        record={"kind":"independent_context_portability_acceptance","created_at":datetime.now(timezone.utc).isoformat(),
                "passed":all(c["passed"] for c in cases),"checks":sum(len(c["checks"]) for c in cases),"cases":cases,
                "classification":"Actual saved exports, independent raw CSV scan, static HTML inspection and simulated source unavailability; no model or browser.",
                "limits":["No tracker identity, detection, calibrated measurement or legal ground truth is established.",
                          "Portable reports contain bounded excerpts and exported thumbnails, not complete CSV or footage.",
                          "Artifact hashes check correspondence, not creator authenticity. Original export fingerprints cannot be independently rechecked without the exports.",
                          "Standalone file-URL visual access remains denied and was not retried or bypassed."]}
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8") as stream:json.dump(record,stream,indent=2)
        print(json.dumps({"passed":record["passed"],"checks":record["checks"],"receipt":str(output),"mismatches":sum(len(c["mismatches"]) for c in cases)}))
        return int(not record["passed"])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print("Context/portability acceptance failed: "+str(exc),file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
