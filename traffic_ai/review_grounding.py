"""Exact source citations and deterministic claim checks; no model or media I/O."""
from copy import deepcopy
import hashlib
import json


def pointer_value(root,pointer):
    # The earlier review format used '/' for the event root; retain that alias.
    if pointer in ("", "/"):return True,root
    if not isinstance(pointer,str) or not pointer.startswith("/"):
        raise ValueError("Invalid source pointer.")
    value=root
    for part in pointer[1:].split("/"):
        part=part.replace("~1","/").replace("~0","~")
        if isinstance(value,dict) and part in value:value=value[part]
        elif isinstance(value,list) and part.isascii() and part.isdecimal() and int(part)<len(value):value=value[int(part)]
        else:return False,None
    return True,value


def citations(root,pointers):
    result=[]
    for pointer in pointers:
        recorded,value=pointer_value(root,pointer)
        ref={"pointer":pointer,"recorded":recorded}
        if recorded:ref["value"]=deepcopy(value)
        result.append(ref)
    return result


def _same(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def validate_grounding(document,sources):
    """Bind every explanation fact to authoritative event/CSV bytes already read.

    This validates representation and citations, not the detector, calibration,
    original footage identity, applicable rules or truth of exporter assertions.
    """
    from .evidence_review import ReviewError,review_event,strict_json
    from .review_tracks import attach_track_context
    files={f["path"]:f for f in document["source"]["files"]}
    def source(name):
        if name not in sources or name not in files:raise ReviewError("Grounding source is missing: "+name)
        data=sources[name]
        if len(data)!=files[name]["bytes"] or hashlib.sha256(data).hexdigest()!=files[name]["sha256"]:
            raise ReviewError("Grounding source fingerprint differs: "+name)
        return data
    expected=[]
    if document["source"]["event_format"]=="jsonl":
        name="motion-incidents.jsonl";data=source(name)
        if data and not data.endswith(b"\n"):raise ReviewError("Grounding stream is incomplete.")
        for number,line in enumerate(data.splitlines(),1):
            if not line.strip():raise ReviewError("Grounding stream contains an empty record.")
            expected.append(review_event(strict_json(line),{"file":name,"line":number,"sha256":hashlib.sha256(line).hexdigest()}))
    else:
        name="motion-incidents.json";summary=strict_json(source(name));events=summary.get("events")
        if not isinstance(events,list) or summary.get("event_count")!=len(events) or summary.get("omitted_from_index",0)!=0:
            raise ReviewError("Grounding needs a complete incident index.")
        expected=[review_event(event,{"file":name,"pointer":f"/events/{i}","sha256":files[name]["sha256"]}) for i,event in enumerate(events)]
    for wanted in expected:
        thumbnail=wanted["event"].get("thumbnail")
        if thumbnail and thumbnail not in files:
            wanted["warnings"].append({"code":"thumbnail_missing","text":"The referenced thumbnail is missing; no replacement was inferred."})
    if "motion-tracks.csv" in files:
        attach_track_context(expected,source("motion-tracks.csv"),files["motion-tracks.csv"])
    if len(expected)!=len(document["events"]):raise ReviewError("Grounded event coverage differs from its source.")
    count=refs=unknown=0
    for actual,wanted in zip(document["events"],expected):
        for key in ("event","evidence_ref","facts","explanation","warnings","review_questions","track_context","stop_context","stop_review"):
            if not _same(actual.get(key),wanted.get(key)):
                raise ReviewError("Review claims or citations differ from recorded evidence: "+key)
        count+=len(actual["facts"])
        refs+=sum(len(f["citations"]) for f in actual["facts"])
        unknown+=sum(not ref["recorded"] for f in actual["facts"] for ref in f["citations"])
    missing={r["event"]["thumbnail"] for r in expected if r["event"].get("thumbnail") and r["event"]["thumbnail"] not in files}
    if "motion-tracks.csv" not in files and any(r.get("stop_review") for r in expected):missing.add("motion-tracks.csv")
    return {"version":1,"state":"source_citations_checked","event_count":len(expected),"fact_count":count,"citation_count":refs,
            "unrecorded_field_citations":unknown,"missing_optional_sources":sorted(missing),
            "limitation":"Checks source correspondence, not measurement accuracy, original-footage identity or a legal finding. Missing optional evidence remains explicitly unresolved."}
