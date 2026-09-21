"""Exact citations and unknown-preserving deterministic explanations; no model."""
from copy import deepcopy

import pytest

from tests.test_review_integration import stopped_event as stopped_event
from tests.test_evidence_review import write_run
from traffic_ai.evidence_review import review_event,load_review,ReviewError
from traffic_ai.review_grounding import pointer_value,validate_grounding
from traffic_ai.review_cli import render_html


def facts(event):
    return {f["id"]:f for f in review_event(event,{"file":"synthetic-fixture.jsonl","line":1})["facts"]}


@pytest.mark.parametrize("name,fields",[
    ("stop_displacement",["position_diameter_bound_m","maximum_position_diameter_m"]),
    ("stop_fitted_speed",["fitted_speed_kmh","maximum_fitted_speed_kmh"]),
    ("stop_window_timing",["start_seconds","end_seconds"]),
])
def test_compound_stop_claim_cites_every_recorded_measurement(stopped_event,name,fields):
    item=facts(stopped_event)[name]
    citations={c["pointer"]:c for c in item.get("citations",[])}
    for field in fields:
        pointer="/measurements/stop_evidence/"+field
        assert citations.get(pointer)=={"pointer":pointer,"recorded":True,"value":stopped_event["measurements"]["stop_evidence"][field]}


def test_stop_interval_legacy_pointer_contains_both_endpoints(stopped_event):
    item=facts(stopped_event)["stop_window_timing"]
    assert item["pointer"]=="/measurements/stop_evidence"
    assert item["value"]==[3.5,4.5]


def test_missing_confirmation_stays_unknown_in_text_and_citation(stopped_event):
    event=deepcopy(stopped_event)
    event["configuration"].pop("calibration_confirmed")
    event["configuration"].pop("real_time_confirmed")
    item=facts(event)["metric_context"]
    assert "unknown (not recorded)" in item["text"]
    assert "calibration confirmed=False" not in item["text"]
    assert item["value"]["calibration_confirmed"] is None
    refs={c["pointer"]:c for c in item["citations"]}
    assert refs["/configuration/calibration_confirmed"]=={"pointer":"/configuration/calibration_confirmed","recorded":False}


def test_explicit_false_confirmation_remains_recorded_false(stopped_event):
    event=deepcopy(stopped_event);event["configuration"]["calibration_confirmed"]=False
    item=facts(event)["metric_context"]
    assert "calibration confirmed=False" in item["text"]
    ref=next(c for c in item.get("citations",[]) if c["pointer"]=="/configuration/calibration_confirmed")
    assert ref["recorded"] is True and ref["value"] is False


def sources_for(run,document):
    return {ref["path"]:(run/ref["path"]).read_bytes() for ref in document["source"]["files"]
            if ref["path"] in ("motion-incidents.jsonl","motion-incidents.json","motion-tracks.csv")}


@pytest.mark.parametrize("mutation",["text","citation","event","coverage","explanation","warning","question"])
def test_added_or_changed_claims_are_rejected_against_source_bytes(tmp_path,stopped_event,mutation):
    run=write_run(tmp_path/"source",[stopped_event]);doc=load_review(run);sources=sources_for(run,doc)
    assert validate_grounding(doc,sources)["state"]=="source_citations_checked"
    if mutation=="text":doc["events"][0]["facts"][0]["text"]="A newly invented confirmed violation"
    elif mutation=="citation":doc["events"][0]["facts"][0]["citations"][0]["value"]="withheld"
    elif mutation=="event":doc["events"][0]["event"]["measurements"]["speed"]=999
    elif mutation=="coverage":doc["events"].clear()
    elif mutation=="warning":doc["events"][0]["warnings"].clear()
    elif mutation=="question":doc["events"][0]["review_questions"].clear()
    else:doc["events"][0]["explanation"].append("An unsupported new claim")
    with pytest.raises(ReviewError,match="differ"):validate_grounding(doc,sources)


@pytest.mark.parametrize("mutation",["missing","corrupt"])
def test_essential_source_missing_or_changed_withholds_grounding(tmp_path,stopped_event,mutation):
    run=write_run(tmp_path/"source",[stopped_event]);doc=load_review(run);sources=sources_for(run,doc)
    if mutation=="missing":sources.pop("motion-incidents.jsonl")
    else:sources["motion-incidents.jsonl"]+=b" "
    with pytest.raises(ReviewError,match="source.*(missing|differs)"):validate_grounding(doc,sources)


def test_complete_json_fallback_is_a_cited_source_not_an_invented_stream(tmp_path,stopped_event):
    run=write_run(tmp_path/"source",[stopped_event]);(run/"motion-incidents.jsonl").unlink()
    doc=load_review(run)
    assert doc["source"]["event_format"]=="json"
    assert validate_grounding(doc,sources_for(run,doc))["event_count"]==1


def test_missing_optional_csv_keeps_exporter_record_but_no_verification(tmp_path,stopped_event):
    run=write_run(tmp_path/"source",[stopped_event]);doc=load_review(run)
    result=validate_grounding(doc,sources_for(run,doc))
    assert "motion-tracks.csv" in result["missing_optional_sources"]
    assert doc["events"][0]["stop_review"]["csv_coverage"]=="unavailable"
    assert doc["events"][0]["event"]==stopped_event


def test_source_pointer_escape_and_missing_differ_from_recorded_null():
    value={"a/b":{"~key":None},"items":[3]}
    assert pointer_value(value,"/a~1b/~0key")== (True,None)
    assert pointer_value(value,"/absent")== (False,None)
    assert pointer_value(value,"/items/0")== (True,3)


def test_html_shows_all_compound_citations_and_missing_fields(tmp_path,stopped_event):
    stopped_event["configuration"].pop("calibration_confirmed")
    doc=load_review(write_run(tmp_path/"source",[stopped_event]))
    page=render_html(doc)
    assert "event:/measurements/stop_evidence/end_seconds" in page
    assert "event:/measurements/stop_evidence/maximum_fitted_speed_kmh" in page
    assert "event:/configuration/calibration_confirmed</code> (not recorded)" in page


def test_csv_claims_are_recomputed_from_verified_rows(tmp_path,stopped_event):
    run=write_run(tmp_path/"source",[stopped_event])
    (run/"motion-tracks.csv").write_text("timestamp_seconds,track_id,speed,speed_unit,speed_status,tracking_state\n"
                                       "3.5,1,0.14,km/h,estimated,observed\n")
    stopped_event["thumbnail"]="missing.png"
    # Recreate both matching event exports without touching the CSV.
    write_run(run,[stopped_event])
    doc=load_review(run);sources=sources_for(run,doc)
    assert validate_grounding(doc,sources)["state"]=="source_citations_checked"
    doc["events"][0]["stop_context"]["samples"][0]["values"]["speed"]="999"
    with pytest.raises(ReviewError,match="stop_context"):validate_grounding(doc,sources)
