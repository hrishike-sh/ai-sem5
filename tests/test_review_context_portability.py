"""Independent CSV joins and detached report checks using synthetic test evidence."""
from copy import deepcopy
import csv
import hashlib
from html.parser import HTMLParser
import io
import json
import shutil

import pytest

from tests.test_evidence_review import incident as incident,write_run,TRACK_HEADER
from tests.test_review_integration import stopped_event as stopped_event
from traffic_ai.evidence_review import load_review,ReviewError
from traffic_ai.review_cli import export_review,validate_review_bundle
from traffic_ai.review_grounding import validate_grounding


@pytest.mark.parametrize("identifier,row_id",[(None,"None"),("",""),("   ","   "),(True,"True"),(4.0,"4.0"),([4],"[4]"),({"id":4},"{'id': 4}")])
def test_unavailable_track_identity_cannot_borrow_named_track_rows(tmp_path,incident,identifier,row_id):
    # Identity faults only; every measurement is copied from the existing fixture.
    incident["track_id"]=identifier
    run=write_run(tmp_path/"source",[incident])
    output=io.StringIO(newline="");writer=csv.writer(output)
    writer.writerow(TRACK_HEADER.strip().split(","))
    writer.writerow([.2,row_id,"12.30","px/s","estimated","observed"])
    (run/"motion-tracks.csv").write_text(output.getvalue())
    item=load_review(run)["events"][0]
    assert item["event"]==incident
    assert item["track_context"]["sample_count"]==0
    assert "track_identity_unavailable" in {w["code"] for w in item["warnings"]}


def test_missing_track_id_is_not_the_named_track_none(tmp_path,incident):
    incident.pop("track_id")
    run=write_run(tmp_path/"source",[incident])
    (run/"motion-tracks.csv").write_text(TRACK_HEADER+"0.2,None,12.30,px/s,estimated,observed\n")
    item=load_review(run)["events"][0]
    assert item["track_context"]["sample_count"]==0
    assert "track_identity_unavailable" in {w["code"] for w in item["warnings"]}


def test_unavailable_stop_track_cannot_claim_matching_csv_proof(tmp_path,stopped_event):
    stopped_event.pop("track_id")
    run=write_run(tmp_path/"source",[stopped_event])
    rows=TRACK_HEADER+"".join(f"{3.5+i/16},None,0.14,km/h,estimated,observed\n" for i in range(17))
    (run/"motion-tracks.csv").write_text(rows)
    doc=export_review(run,tmp_path/"review");item=doc["events"][0]
    assert item["stop_context"]["sample_count"]==0
    assert item["stop_review"]["csv_coverage"]=="incomplete_or_conflicting"
    assert "track_identity_unavailable" in {w["code"] for w in item["warnings"]}
    assert item["event"]==stopped_event


@pytest.mark.parametrize("identifier,row_id",[(4,"4"),("4","4"),("None","None"),("camera-A","camera-A")])
def test_explicit_integer_or_named_track_identity_is_retained(tmp_path,incident,identifier,row_id):
    incident["track_id"]=identifier
    run=write_run(tmp_path/"source",[incident])
    (run/"motion-tracks.csv").write_text(TRACK_HEADER+f"0.2,{row_id},12.30,px/s,estimated,observed\n")
    item=load_review(run)["events"][0]
    assert item["track_context"]["sample_count"]==1
    assert item["track_context"]["samples"][0]["values"]["track_id"]==row_id


def test_reversed_events_and_overlapping_tracks_have_independent_csv_citations(tmp_path,incident):
    # Structural variations of existing synthetic fixtures; no invented measurements.
    early=deepcopy(incident);late=deepcopy(incident)
    late.update(event_id="later-source-first",timestamp_seconds=2.3,evidence_timestamp_seconds=2.3,track_id=99)
    early.update(event_id="earlier-source-second")
    rows=(TRACK_HEADER+"0.2,4,12.30,px/s,estimated,observed\n"
          "0.3,99,30,km/h,estimated,observed\n"
          "1.3,4,,px/s,collecting stable track,ambiguous overlap\n"
          "1.3,99,30,km/h,estimated,observed\n"
          "2.3,4,42,px/s,estimated,observed\n"
          "2.4,4,43,px/s,estimated,observed\n")
    run=write_run(tmp_path/"source",[late,early]);(run/"motion-tracks.csv").write_text(rows)
    doc=export_review(run,tmp_path/"review")
    parsed=list(csv.DictReader(io.StringIO(rows)))
    assert [r["event"] for r in doc["events"]]==[late,early]
    for line,item in enumerate(doc["events"],1):
        event=item["event"];context=item["track_context"]
        lower=max(0,event["timestamp_seconds"]-2);upper=event["evidence_timestamp_seconds"]+1
        expected=[(n,r) for n,r in enumerate(parsed,1) if r["track_id"]==str(event["track_id"]) and lower<=float(r["timestamp_seconds"])<=upper]
        assert [(s["record_number"],s["values"]) for s in context["samples"]]==expected
        raw=(run/"motion-incidents.jsonl").read_bytes().splitlines()[line-1]
        assert item["evidence_ref"]=={"file":"motion-incidents.jsonl","line":line,"sha256":hashlib.sha256(raw).hexdigest()}
    assert doc["events"][0]["track_context"]["unresolved_rows"]==0
    assert doc["events"][1]["track_context"]["unresolved_rows"]==1


class ResourceParser(HTMLParser):
    def __init__(self):super().__init__();self.resources=[];self.tags=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        self.resources.extend(v for k,v in attrs if k in {"src","href","action","poster","srcset","data"})


def test_detached_download_has_events_context_and_image_without_source_access(tmp_path,incident):
    from PIL import Image
    source=tmp_path/"source";bundle=tmp_path/"bundle";detached=tmp_path/"another-machine"
    incident["thumbnail"]="incident-000001.png"
    write_run(source,[incident]);Image.new("RGB",(10,10),"gray").save(source/incident["thumbnail"])
    (source/"motion-tracks.csv").write_text(TRACK_HEADER+"0.2,4,12.30,px/s,estimated,observed\n")
    original=export_review(source,bundle);detached.mkdir()
    for name in ("review.json","review.html","review-complete.json"):shutil.copyfile(bundle/name,detached/name)
    # Remove only files created by this test, never an original user fixture.
    for path in source.iterdir():
        assert path.is_file() and path.parent==tmp_path/"source"
        path.unlink()
    source.rmdir()
    assert validate_review_bundle(detached)["complete"] is True
    doc=json.loads((detached/"review.json").read_text(encoding="utf-8"))
    assert doc==original and doc["events"][0]["event"]==incident
    assert doc["events"][0]["track_context"]["samples"][0]["values"]["speed"]=="12.30"
    page=(detached/"review.html").read_text(encoding="utf-8")
    parsed=ResourceParser();parsed.feed(page)
    assert parsed.resources and all(r.startswith("data:image/png;base64,") for r in parsed.resources)
    assert not ({"script","iframe","link","object","embed","base","form"}&set(parsed.tags))
    assert "Original event (unchanged)" in page and "Track CSV excerpt" in page
    assert "12.30" in page and "motion-tracks.csv" in page
    # A portable snapshot remains readable; source-grounding cannot be rerun without exports.
    with pytest.raises(ReviewError,match="source is missing"):validate_grounding(doc,{})


def test_detached_report_keeps_excerpt_omissions_explicit(tmp_path,incident):
    run=write_run(tmp_path/"source",[incident])
    rows=TRACK_HEADER+"".join(f"{i/100},4,0,km/h,estimated,observed\n" for i in range(201))
    (run/"motion-tracks.csv").write_text(rows)
    doc=export_review(run,tmp_path/"review")
    context=doc["events"][0]["track_context"]
    assert (context["sample_count"],context["shown_samples"],context["omitted_samples"])==(201,120,81)
    assert "81 records omitted" in (tmp_path/"review/review.html").read_text(encoding="utf-8")
