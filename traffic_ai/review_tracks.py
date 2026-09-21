"""Bounded, lossless CSV excerpts for incident review; no speed/stop inference."""
from collections import defaultdict, deque
import csv
import io
import math

from .evidence_review import ReviewError, _finite
from .review_grounding import citations


def _track_key(value):
    # CSV has text IDs. Coercing null/bool/containers would borrow a different
    # named track's rows (for example an absent ID becoming the literal 'None').
    if type(value) is int:return str(value)
    if isinstance(value,str) and value.strip():return value
    return None


def attach_track_context(reviews, data, file_ref):
    if not reviews:
        return
    groups = defaultdict(list)

    def add_window(review, field, start, end):
        state = {"review": review, "field": field, "start": start, "end": end, "count": 0,
                 "first": [], "last": deque(maxlen=60), "observed": 0, "unresolved": 0,
                 "first_stamp": None, "last_stamp": None, "duplicates": 0}
        groups[_track_key(review["event"].get("track_id"))].append(state)

    for review in reviews:
        event = review["event"]
        if _track_key(event.get("track_id")) is None:
            review["warnings"].append({"code":"track_identity_unavailable",
                "text":"Track identity is missing or is not an explicit integer or nonblank string; no CSV rows were attributed to this event. Source values and status are unchanged."})
        stamp = event["timestamp_seconds"]
        frame_stamp = event.get("evidence_timestamp_seconds", stamp)
        if not _finite(frame_stamp) or frame_stamp < stamp:
            frame_stamp = stamp
        start, end = max(0., stamp-2.), frame_stamp+1.
        add_window(review, "track_context", start, end)
        stop_window = review.get("stop_review", {}).get("csv_window_seconds")
        if stop_window:
            add_window(review, "stop_context", *stop_window)
    for group in groups.values():
        group.sort(key=lambda state: state["start"])
    active, offsets = defaultdict(list), defaultdict(int)
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        required = {"timestamp_seconds", "track_id", "speed", "speed_unit", "speed_status", "tracking_state"}
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields) or not required <= set(fields):
            raise ReviewError("Track CSV is missing required or has duplicate columns.")
        previous, matches = -1., 0
        for record, row in enumerate(reader, 1):
            if record > 1000000:
                raise ReviewError("Track CSV exceeds the one-million-record review limit.")
            if None in row or any(v is None for v in row.values()):
                raise ReviewError("Track CSV record has an inconsistent column count.")
            try:
                stamp = float(row["timestamp_seconds"])
            except ValueError as exc:
                raise ReviewError("Track CSV contains a nonnumeric timestamp.") from exc
            if not math.isfinite(stamp) or stamp < 0 or stamp < previous:
                raise ReviewError("Track CSV timestamps must be finite, nonnegative and ordered.")
            previous = stamp
            key = row["track_id"]
            if key not in groups:
                continue
            pending = groups.get(key, [])
            index = offsets[key]
            while index < len(pending) and pending[index]["start"]-1e-6 <= stamp:
                active[key].append(pending[index])
                index += 1
            offsets[key] = index
            active[key] = [s for s in active[key] if stamp <= s["end"]+1e-6]
            for state in active[key]:
                matches += 1
                if matches > 1000000:
                    raise ReviewError("Track excerpt join exceeds the one-million-match review limit.")
                sample = {"record_number": record, "line_end": reader.line_num, "values": dict(row)}
                state["count"] += 1
                if state["first_stamp"] is None:
                    state["first_stamp"] = stamp
                if state["last_stamp"] is not None and abs(stamp-state["last_stamp"]) < 1e-9:
                    state["duplicates"] += 1
                state["last_stamp"] = stamp
                if len(state["first"]) < 60:
                    state["first"].append(sample)
                else:
                    state["last"].append(sample)
                state["observed" if row["tracking_state"] == "observed" else "unresolved"] += 1
    except (csv.Error, UnicodeError) as exc:
        raise ReviewError("Invalid track CSV: " + str(exc)) from exc
    for group in groups.values():
        for state in group:
            review = state["review"]
            samples = state["first"] + list(state["last"])
            count = state["count"]
            context = {"source": dict(file_ref), "requested_window_seconds": [state["start"], state["end"]],
                       "sample_count": count, "shown_samples": len(samples), "omitted_samples": count-len(samples),
                       "observed_rows": state["observed"], "unresolved_rows": state["unresolved"],
                       "duplicate_timestamps": state["duplicates"],
                       "first_timestamp_seconds": state["first_stamp"], "last_timestamp_seconds": state["last_stamp"],
                       "samples": samples,
                       "policy": "First 60 and last 60 matching records. Event context spans 2 seconds before the event to 1 second after its evidence frame; stop context uses the recorded stop interval. Raw values retained.",
                       "limitation": "An excerpt does not establish full approach history, a completed stop or identity continuity. Check the complete source CSV and footage."}
            field = state["field"]
            review[field] = context
            text = (f"{'Stop-window' if field == 'stop_context' else 'Track'} CSV context: {count} matching records; {state['observed']} marked observed and "
                    f"{state['unresolved']} with another tracking state. {count-len(samples)} records omitted from the displayed excerpt.")
            review["facts"].append({"id": field, "pointer": "/"+field, "pointer_base": "review",
                                    "value": {k: context[k] for k in ("sample_count", "observed_rows", "unresolved_rows", "omitted_samples")},
                                    "text": text, "source_ref": dict(file_ref),
                                    "citations":citations(review,["/"+field+"/"+k for k in ("sample_count","observed_rows","unresolved_rows","omitted_samples")])})
            review["explanation"].append(text)
            if not count:
                review["warnings"].append({"code": "no_track_context", "text": "No matching track CSV samples were found in the requested event window."})
            if state["unresolved"]:
                review["warnings"].append({"code": "unresolved_track_context", "text": "The CSV excerpt contains unobserved or uncertain tracking states; do not bridge their history."})
            if state["duplicates"]:
                review["warnings"].append({"code": "duplicate_track_timestamps", "text": "Repeated track timestamps are not independent observations."})
            if field == "stop_context":
                proof = review["event"]["measurements"]["stop_evidence"]
                matching = (count == proof.get("observed_samples") and not state["unresolved"] and
                            not state["duplicates"] and count > 0 and
                            abs(state["first_stamp"]-state["start"]) <= 1e-6 and
                            abs(state["last_stamp"]-state["end"]) <= 1e-6)
                review["stop_review"]["csv_coverage"] = "count_and_states_match" if matching else "incomplete_or_conflicting"
                review["warnings"] = [w for w in review["warnings"] if w["code"] != "stop_csv_unavailable"]
                if not matching:
                    review["warnings"].append({"code": "stop_csv_evidence_mismatch", "text": "Stop-window CSV count, endpoints or tracking states disagree with the recorded proof; do not treat the proof as independently verified."})
