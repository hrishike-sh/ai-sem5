"""Explain recorded stop windows without recomputing stops or changing verdicts."""
from copy import deepcopy
import math

from .evidence_review import _finite
from .review_grounding import citations


def attach_stop_review(review):
    event = review["event"]
    measurements, config = event.get("measurements", {}), event.get("configuration", {})
    proof = measurements.get("stop_evidence")
    claimed = measurements.get("stopped_in_approach")
    if measurements.get("condition") != "stop_required" and proof is None and claimed is not True:
        return
    result = {"state": "missing", "csv_coverage": "unavailable", "issues": [],
              "limitation": "Recorded low motion does not prove exactly stationary movement, a legal stop or full approach coverage. Source status is unchanged."}
    review["stop_review"] = result

    def warn(code, text):
        result["issues"].append(code)
        review["warnings"].append({"code": code, "text": text})

    def fact(key, field, value, text, source_fields=None):
        pointer="/measurements/stop_evidence"+("/"+field if field else "")
        review["facts"].append({"id": key, "pointer": pointer,
                                "value": deepcopy(value), "text": text,
                                "citations":citations(event,["/measurements/stop_evidence/"+f for f in (source_fields or [field])])})
        review["explanation"].append(text)

    if proof is None:
        warn("stop_record_missing", "No stop-window record was exported. Do not infer a completed stop from its absence or from a single speed value.")
        return
    result["state"] = "inconsistent"
    if not isinstance(proof, dict):
        warn("stop_record_malformed", "The optional stop-window record is not an object; its source value is retained for audit.")
        return
    if claimed is not True:
        warn("stop_flag_conflict", "A stop-window record exists but stopped_in_approach is not true; inspect the source export.")
    numeric = ("start_seconds", "end_seconds", "duration_seconds", "position_diameter_bound_m",
               "maximum_position_diameter_m", "fitted_speed_kmh", "maximum_fitted_speed_kmh")
    if any(not _finite(proof.get(k)) or proof[k] < 0 for k in numeric):
        warn("stop_record_malformed", "Stop-window time, displacement and speed fields must be finite nonnegative numbers.")
        return
    start, end, duration = (proof[k] for k in numeric[:3])
    frame_time = event.get("evidence_timestamp_seconds", event["timestamp_seconds"])
    if (end < start or duration < 1-1e-6 or not math.isclose(duration, end-start, abs_tol=1e-6, rel_tol=1e-6)
            or not _finite(frame_time) or end > frame_time+1e-6):
        warn("stop_interval_inconsistent", "Recorded stop duration or bounds are inconsistent, too short, or later than the evidence frame.")
    samples = proof.get("observed_samples")
    if type(samples) is not int or not 8 <= samples <= 256:
        warn("stop_samples_inconsistent", "The current stop exporter requires 8 to 256 real observations; the recorded count is unsupported.")
    if (not math.isclose(proof["maximum_position_diameter_m"], 2/3.6, abs_tol=1e-6) or
            not math.isclose(proof["maximum_fitted_speed_kmh"], 2., abs_tol=1e-6)):
        warn("stop_policy_unsupported", "The recorded stop policy differs from the current 2 km/h and 0.555556 m low-motion policy; review its provenance.")
    if (proof["position_diameter_bound_m"] > proof["maximum_position_diameter_m"]+1e-6 or
            proof["fitted_speed_kmh"] > proof["maximum_fitted_speed_kmh"]+1e-6):
        warn("stop_motion_bounds_exceeded", "Recorded displacement or fitted speed exceeds its own stop-window policy bound.")
    calibration = event.get("calibration")
    if (not calibration or config.get("calibration_confirmed") is not True or
            config.get("real_time_confirmed") is not True or measurements.get("speed_unit") != "km/h"):
        warn("stop_metric_context_missing", "Recorded stop metrics lack confirmed calibration, real-time timing or metric units.")
    if (calibration and calibration.get("source_revision") and event.get("source_revision") and
            calibration["source_revision"] != event["source_revision"]):
        warn("stop_calibration_source_mismatch", "Stop calibration and event source revisions disagree; this proof cannot be treated as verified.")
    if not isinstance(proof.get("method"), str) or not proof["method"].strip():
        warn("stop_method_missing", "The stop-window method was not recorded.")
    if not result["issues"]:
        result["state"] = "internally_consistent"
    # Even consistent records are citations of source measurements, not a new finding.
    fact("stop_duration", "duration_seconds", duration, f"Exporter stop-window duration: {duration:g} seconds.")
    fact("stop_samples", "observed_samples", samples, f"Exporter stop-window observation count: {samples}.")
    fact("stop_displacement", "position_diameter_bound_m", proof["position_diameter_bound_m"],
         f"Exporter stop-window displacement bound: {proof['position_diameter_bound_m']:g} m; recorded maximum {proof['maximum_position_diameter_m']:g} m.",
         ["position_diameter_bound_m","maximum_position_diameter_m"])
    fact("stop_fitted_speed", "fitted_speed_kmh", proof["fitted_speed_kmh"],
         f"Exporter stop-window fitted speed: {proof['fitted_speed_kmh']:g} km/h; recorded maximum {proof['maximum_fitted_speed_kmh']:g} km/h.",
         ["fitted_speed_kmh","maximum_fitted_speed_kmh"])
    fact("stop_window_timing", "", [start, end],
         f"Exporter stop window: {start:g} to {end:g} seconds. Keep its fitted speed separate from the speed field at the event observation.",
         ["start_seconds","end_seconds"])
    if _finite(frame_time) and 0 <= start <= end <= frame_time and end-start <= 300:
        result["csv_window_seconds"] = [start, end]
    else:
        warn("stop_csv_window_unavailable", "The stop CSV interval is invalid or exceeds the bounded 300-second excerpt window.")
    warn_text = "Stop-window CSV coverage is unavailable; recorded metric consistency is not independent measurement verification."
    review["warnings"].append({"code": "stop_csv_unavailable", "text": warn_text})
