"""Read-only, evidence-cited review of the classical motion incident exports.

This module deliberately imports no detector, worker, or language-model runtime.
Source statuses and measurements are records, never conclusions recomputed here.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .review_grounding import citations


STATUSES = {"candidate", "withheld", "observation"}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
QUESTIONS = {
    "track": "Check identity continuity, occlusion, clipping and camera stability around the evidence frame.",
    "scene": "Check that the configured line or zone and supplied condition apply to this road user.",
    "calibration": "Check measured road dimensions, calibration region and real-time recording before interpreting km/h.",
    "timing": "Inspect both ends of the crossing bracket and any condition transition inside it.",
    "stop": "Inspect the complete approach history; a still thumbnail cannot establish a completed stop.",
    "synthetic": "Keep declared simulation scale separate from measured road truth and real-traffic accuracy.",
    "source": "Locate the original footage and verify its content hash before relying on the evidence.",
}
LIMITATIONS = [
    "Source statuses are preserved: candidates require review, withheld means unresolved, and observations do not establish a violation.",
    "No new events, calibrated speeds or legal findings are inferred by this reviewer.",
    "Detector scores are not probabilities that an incident occurred. A policy speed guard is not a validated error bound or legal tolerance.",
    "Zero exported events does not establish that no incidents occurred; consult detector coverage and excluded frames.",
]


class ReviewError(ValueError):
    """An export cannot safely be represented as a complete evidence review."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def strict_json(data):
    def invalid(value):
        raise ReviewError("Non-finite JSON value: " + value)
    try:
        result = json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid)
        _check_finite(result)
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ReviewError("Invalid evidence JSON: " + str(exc)) from exc


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ReviewError("Evidence contains a non-finite number.")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def _read(root, name, manifest, *, required=False, limit=MAX_FILE_BYTES):
    path = root / name
    # No file referenced by an export may escape its run, including symlinks.
    if path.resolve().parent != root:
        raise ReviewError("Evidence path escapes the run: " + name)
    if not path.is_file():
        if required:
            raise ReviewError("Missing evidence file: " + name)
        return None
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ReviewError("Evidence file exceeds the review limit: " + name)
    manifest[name] = {"path": name, "sha256": digest(data), "bytes": len(data)}
    return data


def validate_event(event):
    if not isinstance(event, dict):
        raise ReviewError("Each incident must be a JSON object.")
    for key in ("event_id", "type", "reason"):
        if not isinstance(event.get(key), str) or not event[key] or len(event[key]) > 16000:
            raise ReviewError("Invalid incident " + key)
    if not isinstance(event.get("status"), str) or event["status"] not in STATUSES:
        raise ReviewError("Unsupported incident status; refusing to reinterpret it.")
    if not _finite(event.get("timestamp_seconds")) or event["timestamp_seconds"] < 0:
        raise ReviewError("Incident timestamp must be finite nonnegative seconds.")
    if "synthetic" in event and type(event["synthetic"]) is not bool:
        raise ReviewError("Incident synthetic label must be a boolean.")
    for key in ("measurements", "quality", "configuration", "coordinates"):
        if key in event and not isinstance(event[key], dict):
            raise ReviewError("Incident " + key + " must be an object.")
    if event.get("calibration") is not None and not isinstance(event["calibration"], dict):
        raise ReviewError("Incident calibration must be an object or null.")
    if event.get("source_revision") is not None and not isinstance(event["source_revision"], str):
        raise ReviewError("Incident source_revision must be a string or null.")
    for key in ("calibration_confirmed", "real_time_confirmed", "synthetic"):
        config = event.get("configuration", {})
        if key in config and type(config[key]) is not bool:
            raise ReviewError("Incident configuration " + key + " must be a boolean.")
    _check_finite(event)


def review_event(event, reference):
    """Return deterministic facts and review prompts, with a lossless source object."""
    validate_event(event)
    event = deepcopy(event)
    facts, warnings, questions = [], [], []

    def fact(key, pointer, value, text, source_pointers=None):
        facts.append({"id": key, "pointer": pointer, "value": deepcopy(value), "text": text,
                      "citations":citations(event,source_pointers or [pointer])})

    def warn(code, text):
        warnings.append({"code": code, "text": text})

    def question(key):
        if key not in questions:
            questions.append(key)

    status = event["status"]
    meanings = {"candidate": "The exporter recorded a review candidate; human review is required.",
                "withheld": "The exporter withheld assessment; the evidence is unresolved.",
                "observation": "The exporter recorded an observation; it does not establish a violation."}
    fact("status", "/status", status, meanings[status])
    fact("timestamp", "/timestamp_seconds", event["timestamp_seconds"],
         f"Event time: {event['timestamp_seconds']} seconds.")
    fact("type", "/type", event["type"], "Exported event type: " + event["type"] + ".")
    fact("reason", "/reason", event["reason"], "Exporter reason (quoted evidence): " + event["reason"])
    synthetic = event.get("synthetic")
    if synthetic is True:
        fact("synthetic", "/synthetic", True, "SYNTHETIC evidence; declared scale and conditions do not establish real-world accuracy.")
        question("synthetic")
    elif synthetic is None:
        warn("synthetic_unknown", "Synthetic provenance was not recorded; it must not be assumed to be real footage.")
    else:
        fact("synthetic", "/synthetic", False, "The exporter labels this evidence as non-synthetic; provenance still requires review.")
    config = event.get("configuration", {})
    if "synthetic" in config and synthetic is not None and config["synthetic"] != synthetic:
        warn("synthetic_conflict", "Event and configuration synthetic labels disagree; preserve both and resolve provenance.")
    revision = event.get("source_revision")
    if revision:
        fact("source", "/source_revision", revision, "Recorded source revision (not independently verified as a footage content hash): " + revision)
    else:
        warn("missing_source_revision", "The export lacks a source revision; media identity remains unknown here.")
        question("source")
    quality, measurements = event.get("quality", {}), event.get("measurements", {})
    question("track")
    state = quality.get("tracking_state")
    if state != "observed":
        warn("track_unresolved", "Tracking state is missing or is not observed; inspect continuity before relying on this event.")
    if "detector_confidence" in quality:
        score = quality["detector_confidence"]
        fact("detector_score", "/quality/detector_confidence", score,
             f"Recorded detector score: {score}; this is not an incident probability.")
        if not _finite(score) or not 0.35 <= score <= 1:
            warn("detector_score_unresolved", "Detector score is missing, invalid or below the exporter's minimum review score.")
    speed = measurements.get("speed")
    unit = measurements.get("speed_unit")
    metric = (bool(event.get("calibration")) and config.get("calibration_confirmed") is True
              and config.get("real_time_confirmed") is True and quality.get("speed_status") == "estimated"
              and unit == "km/h" and _finite(speed) and speed >= 0)
    calibration = event.get("calibration") or {}
    if calibration.get("source_revision") and revision and calibration["source_revision"] != revision:
        metric = False
        warn("calibration_source_mismatch", "Calibration and event source revisions disagree; confirmed flags cannot establish applicability.")
    if speed is not None:
        def recorded_text(container,key):
            if key not in container:return "unknown (not recorded)"
            if container[key] is None:return "unknown (recorded null)"
            return str(container[key])
        context = {"calibration": event.get("calibration"), "calibration_confirmed": config.get("calibration_confirmed"),
                   "real_time_confirmed": config.get("real_time_confirmed"), "speed_status": quality.get("speed_status")}
        fact("metric_context", "/", context,
             f"Recorded metric context: calibration present={bool(event.get('calibration')) if 'calibration' in event else 'unknown (not recorded)'}, "
             f"calibration confirmed={recorded_text(config,'calibration_confirmed')}, "
             f"real-time confirmed={recorded_text(config,'real_time_confirmed')}, "
             f"speed status={recorded_text(quality,'speed_status')}.",
             ["/calibration","/configuration/calibration_confirmed","/configuration/real_time_confirmed","/quality/speed_status"])
        fact("speed", "/measurements", {"speed": speed, "speed_unit": unit},
             f"Recorded speed field: {speed} {unit or '(unit missing)'}. " +
             ("The export records synthetic scale and timing; this is not measured real-road truth."
              if metric and synthetic is True else
              "The export records confirmed calibration and real-time timing; verify their applicability."
              if metric else "This reviewer cannot establish a calibrated road speed from this field."),
             ["/measurements/speed","/measurements/speed_unit","/calibration","/configuration/calibration_confirmed",
              "/configuration/real_time_confirmed","/quality/speed_status","/synthetic","/source_revision","/calibration/source_revision"])
        if not metric:
            warn("metric_speed_unverified", "No calibrated speed interpretation: require km/h, reliable estimate, confirmed measured calibration and real-time recording.")
            question("calibration")
    if unit == "px/s":
        warn("pixel_speed", "Pixel speed is image motion; it cannot establish km/h or a speeding finding.")
    if event["type"] in ("speeding", "speed_assessment"):
        question("calibration")
        if not metric:
            warn("speed_evidence_incomplete", "The speed assessment lacks required calibrated evidence. The original status is retained for audit.")
        if event["type"] == "speeding" and status == "candidate":
            limit, guard = measurements.get("threshold_kmh"), measurements.get("measurement_guard_kmh")
            duration, samples = measurements.get("sustained_seconds"), measurements.get("observed_samples")
            if (not _finite(limit) or limit <= 0 or not _finite(guard) or guard < 0 or
                    not _finite(duration) or duration < .5 or type(samples) is not int or samples < 3 or
                    not _finite(speed) or (_finite(limit) and _finite(guard) and speed <= limit+guard)):
                warn("speed_candidate_evidence_inconsistent", "The source candidate lacks consistent threshold, guard, duration or sample-count evidence; inspect the original export.")
    condition = measurements.get("condition")
    if condition is not None:
        fact("condition", "/measurements/condition", condition,
             f"Recorded applicable condition: {condition}. Conditions come from scene configuration, not LLM signal recognition.")
        question("scene")
        if condition == "unknown":
            warn("condition_unknown", "The applicable condition is unknown or ambiguous; it cannot support a new conclusion.")
    if event["type"] in ("line_crossing", "zone_entry"):
        question("scene")
    bracket = measurements.get("crossing_bracket_seconds")
    if bracket is not None:
        fact("crossing_bracket", "/measurements/crossing_bracket_seconds", bracket,
             f"Recorded crossing observation bracket: {bracket} seconds; event time may be interpolated.")
        question("timing")
        if (not isinstance(bracket, list) or len(bracket) != 2 or
                any(not _finite(v) for v in bracket) or not 0 <= bracket[0] <= event["timestamp_seconds"] <= bracket[1]):
            warn("invalid_crossing_bracket", "The crossing time is inconsistent with its recorded bracket; inspect the source export.")
    if condition == "stop_required" or "stopped_in_approach" in measurements:
        question("stop")
    if condition == "stop_required":
        warn("stop_history_requires_review", "This event alone does not contain full approach history; verify the complete track CSV and footage before interpreting the stop assessment.")
    for key in ("threshold_kmh", "measurement_guard_kmh", "sustained_seconds", "observed_samples", "stopped_in_approach"):
        if key in measurements:
            fact(key, "/measurements/" + key, measurements[key], f"Recorded {key}: {measurements[key]}.")
    if "evidence_timestamp_seconds" in event:
        fact("evidence_timestamp", "/evidence_timestamp_seconds", event["evidence_timestamp_seconds"],
             f"Evidence frame time: {event['evidence_timestamp_seconds']} seconds; this may differ from interpolated event time.")
    reviewed = {"event": event, "evidence_ref": deepcopy(reference), "facts": facts, "warnings": warnings,
            "review_questions": [{"id": key, "text": QUESTIONS[key]} for key in questions],
            "explanation": [f["text"] for f in facts], "language_model": {"state": "not_requested"}}
    from .review_stop import attach_stop_review
    attach_stop_review(reviewed)
    return reviewed


def load_review(run, *, max_events=5000):
    """Read a completed run. Prefer JSONL; reject a truncated JSON index fallback."""
    if type(max_events) is not int or not 1 <= max_events <= 100000:
        raise ReviewError("max_events must be between 1 and 100000.")
    root = Path(run).resolve()
    if not root.is_dir():
        raise ReviewError("Run directory does not exist.")
    manifest, warnings = {}, []
    status_data = _read(root, "status.json", manifest, required=True)
    status = strict_json(status_data)
    if not isinstance(status, dict) or status.get("state") != "complete":
        raise ReviewError("Only a completed run can be reviewed; active or partial exports are not accepted.")
    summary_data = _read(root, "motion-incidents.json", manifest)
    summary = strict_json(summary_data) if summary_data is not None else {}
    if not isinstance(summary, dict):
        raise ReviewError("Incident summary must be an object.")
    stream = _read(root, "motion-incidents.jsonl", manifest)
    records = []
    if stream is not None:
        name = "motion-incidents.jsonl"
        if stream and not stream.endswith(b"\n"):
            raise ReviewError("Incident JSONL is not newline-terminated; it may be a partial write.")
        for line_no, line in enumerate(stream.splitlines(), 1):
            if not line.strip() or len(line) > MAX_LINE_BYTES:
                raise ReviewError("Empty or oversized incident line: " + str(line_no))
            if len(records) >= max_events:
                raise ReviewError("Event limit exceeded; increase max_events explicitly. No partial review was created.")
            records.append((strict_json(line), {"file": name, "line": line_no, "sha256": digest(line)}))
    else:
        name = "motion-incidents.json"
        events = summary.get("events")
        count = summary.get("event_count")
        if (not isinstance(events, list) or type(count) is not int or count != len(events)
                or summary.get("omitted_from_index", 0) != 0):
            raise ReviewError("Complete JSONL stream missing and JSON index is incomplete; CSV alone lacks the required evidence.")
        if len(events) > max_events:
            raise ReviewError("Event limit exceeded; no partial review was created.")
        records = [(e, {"file": name, "pointer": f"/events/{i}", "sha256": manifest[name]["sha256"]})
                   for i, e in enumerate(events)]
    if "event_count" in summary and (type(summary["event_count"]) is not int or summary["event_count"] != len(records)):
        raise ReviewError("Incident summary count disagrees with the full stream.")
    if stream is not None and "events" in summary:
        indexed = summary["events"]
        if (not isinstance(indexed, list) or indexed != [record[0] for record in records[:len(indexed)]]):
            raise ReviewError("Incident summary index disagrees with the full stream.")
    seen, reviews = set(), []
    for event, reference in records:
        validate_event(event)
        if event["event_id"] in seen:
            raise ReviewError("Duplicate event ID: " + event["event_id"])
        seen.add(event["event_id"])
        reviewed = review_event(event, reference)
        thumbnail = event.get("thumbnail")
        if thumbnail is not None:
            if (not isinstance(thumbnail, str) or not thumbnail or "\\" in thumbnail or "/" in thumbnail
                    or Path(thumbnail).suffix.lower() != ".png" or ":" in thumbnail):
                raise ReviewError("Unsafe thumbnail reference.")
            data = _read(root, thumbnail, manifest, limit=8 * 1024 * 1024)
            if data is None:
                reviewed["warnings"].append({"code": "thumbnail_missing", "text": "The referenced thumbnail is missing; no replacement was inferred."})
            else:
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ReviewError("Evidence thumbnail is not a PNG.")
                reviewed["thumbnail_ref"] = deepcopy(manifest[thumbnail])
        reviews.append(reviewed)
    counts = dict(Counter(item["event"]["status"] for item in reviews))
    if "counts" in summary:
        recorded = summary["counts"]
        if (not isinstance(recorded, dict) or set(recorded) - STATUSES or
                any(type(v) is not int or v < 0 for v in recorded.values()) or
                any(recorded.get(key, 0) != counts.get(key, 0) for key in STATUSES)):
            raise ReviewError("Incident status counts disagree with the full stream.")
    for key in ("invalid_reason", "excluded_geometry_frames", "evicted_track_states"):
        if summary.get(key):
            warnings.append({"code": key, "text": f"Exporter recorded {key}: {summary[key]}"})
    # report/CSV are hashed supporting exports, never substitutes for full event evidence.
    for name in ("report.json", "motion-incidents.csv"):
        _read(root, name, manifest)
    track_data = _read(root, "motion-tracks.csv", manifest)
    if track_data is not None:
        from .review_tracks import attach_track_context
        attach_track_context(reviews, track_data, manifest["motion-tracks.csv"])
    revisions = sorted({r["event"]["source_revision"] for r in reviews if r["event"].get("source_revision")})
    if len(revisions) > 1:
        warnings.append({"code": "mixed_source_revisions", "text": "The event stream references multiple source revisions; verify that it is one coherent run."})
    return {"schema_version": 1, "kind": "traffic_evidence_review",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {"run_id": root.name, "event_format": "jsonl" if stream is not None else "json",
                       "complete": True, "files": list(manifest.values()), "source_revisions": revisions},
            "summary": {"event_count": len(reviews), "statuses": {key: counts.get(key, 0) for key in sorted(STATUSES)},
                        "synthetic_events": sum(r["event"].get("synthetic") is True for r in reviews),
                        "unknown_provenance_events": sum("synthetic" not in r["event"] for r in reviews)},
            "exporter_summary": deepcopy(summary), "warnings": warnings,
            "limitations": list(LIMITATIONS), "events": reviews}
