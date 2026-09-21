"""Cancelled-source recovery is a new review, never a partial-success report."""
import hashlib
import json
import urllib.request

import pytest

from tests.test_review_integration import stopped_event as stopped_event
from tests.test_evidence_review import write_run
from traffic_ai.evidence_review import ReviewError
from traffic_ai.review_cli import export_review,validate_review_bundle


def test_cancelled_source_with_valid_events_requires_completion_before_retry(tmp_path,stopped_event,monkeypatch):
    source=write_run(tmp_path/"source",[stopped_event]);output=tmp_path/"review"
    complete_status=(source/"status.json").read_bytes()
    event_bytes=(source/"motion-incidents.jsonl").read_bytes()
    event_hash=hashlib.sha256(event_bytes).hexdigest()
    (source/"status.json").write_text(json.dumps({"state":"cancelled"}))
    def no_provider(*args,**kwargs):raise AssertionError("Deterministic retry must not call a model/provider")
    monkeypatch.setattr(urllib.request.OpenerDirector,"open",no_provider)
    with pytest.raises(ReviewError,match="completed run"):
        export_review(source,output)
    assert not output.exists() and (source/"motion-incidents.jsonl").read_bytes()==event_bytes
    # Test fixture completion only: a real cancelled job must be processed again,
    # never have its status edited to manufacture completion.
    (source/"status.json").write_bytes(complete_status)
    document=export_review(source,output)
    assert validate_review_bundle(output)["complete"] is True
    assert document["events"][0]["event"]==stopped_event
    assert document["events"][0]["language_model"]["state"]=="not_requested"
    assert hashlib.sha256((source/"motion-incidents.jsonl").read_bytes()).hexdigest()==event_hash
