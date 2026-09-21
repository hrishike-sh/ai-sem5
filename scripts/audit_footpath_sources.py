"""Audit retained footpath raster/geometry agreement without model inference.

Source masks are model output, not independent semantic ground truth. Original
rule decisions are never recomputed or changed. Missing camera stability stays
unknown; grid-pixel disagreement does not imply an observed traffic error.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image

from traffic_ai.contracts import Detection
from traffic_ai.footpath_support import RasterFootpathSupport
from traffic_ai.geometry import point_in_polygon
from traffic_ai.rules import SceneEvidence


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_run(run):
    run = Path(run).resolve()
    names = ("status.json", "report.json", "coordinates.json", "scene_features.zip", "detections.jsonl")
    protected = {name: sha(run / name) for name in names}
    if json.loads((run / "status.json").read_text())["state"] != "complete":
        raise ValueError("Only complete retained runs can be audited")
    report = json.loads((run / "report.json").read_text())
    width, height = json.loads((run / "coordinates.json").read_text())["processed_size"]
    if type(width) is not int or type(height) is not int or not 0 < width * height <= 16_000_000:
        raise ValueError("Invalid or excessive processed image dimensions")
    scenes, scene_receipts = {}, []
    with zipfile.ZipFile(run / "scene_features.zip") as archive:
        for name in archive.namelist():
            if not re.fullmatch(r"[0-9]+/scene\.json", name):
                continue
            raw = archive.read(name)
            scene = json.loads(raw)
            masks, mask_hashes = {}, {}
            for key in ("road", "sidewalk"):
                member = name.rsplit("/", 1)[0] + "/" + key + ".png"
                content = archive.read(member)
                with Image.open(io.BytesIO(content)) as image:
                    if image.size != (width, height) or image.mode not in ("L", "1"):
                        raise ValueError("Mask dimensions/mode do not match processed coordinates")
                    masks[key] = np.array(image)
                mask_hashes[key] = hashlib.sha256(content).hexdigest()
            identity = (scene["scene_id"], scene["revision"])
            if identity in scenes:
                raise ValueError("Duplicate archived scene identity")
            gate = RasterFootpathSupport(masks["road"], masks["sidewalk"], frame_shape=(height, width),
                                         scene_id=identity[0], revision=identity[1], scores=scene.get("segmentation", {}))
            if gate.errors:
                raise ValueError("Source mask failed binary/coordinate validation")
            polygon_mask = np.zeros((height, width), np.uint8)
            for polygon in scene.get("footpath_polygons", []):
                cv2.fillPoly(polygon_mask, [np.asarray(polygon, np.int32)], 1)
            contours, hierarchy = cv2.findContours(masks["sidewalk"].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            holes = sum(int(row[3] >= 0) for row in hierarchy[0]) if hierarchy is not None else 0
            summary = {"scene_id": identity[0], "revision": identity[1], "scene_member": name,
                       "scene_json_sha256": hashlib.sha256(raw).hexdigest(), "mask_png_sha256": mask_hashes,
                       "sidewalk_holes": holes, "polygon_only_grid_pixels": int(np.count_nonzero((polygon_mask > 0) & (masks["sidewalk"] == 0))),
                       "support_source": gate.source_snapshot()}
            scenes[identity] = (gate, scene, masks)
            scene_receipts.append(summary)
    frames, missing_camera, contacts, centre_mismatch = 0, 0, 0, 0
    spatial, reasons, gate_status, stored_verdict_support = Counter(), Counter(), Counter(), Counter()
    for raw in (run / "detections.jsonl").read_text().splitlines():
        if not raw.strip():
            continue
        frame = json.loads(raw)
        frames += 1
        recorded_stability = type(frame.get("camera_stable")) is bool
        missing_camera += int(not recorded_stability)
        entry = scenes.get((frame.get("scene_id"), frame.get("scene_revision")))
        for value in frame.get("detections", []):
            if value.get("label") not in ("motorcycle", "bicycle"):
                continue
            contacts += 1
            if entry is None:
                reasons["archived_scene_missing"] += 1
                gate_status["withheld"] += 1
                continue
            gate, scene, masks = entry
            vehicle = Detection(**value)
            support = gate.spatial_support(vehicle, scene.get("footpath_polygons", []))
            spatial[support["status"]] += 1
            reasons[support["reason"]] += 1
            matching = [d for d in frame.get("decisions", []) if d.get("rule") == "footpath_riding"
                        and d.get("track_id") == vehicle.track_id and tuple(d.get("box") or ()) == tuple(vehicle.box)]
            if len(matching) == 1:
                stored_verdict_support[str(matching[0].get("verdict")) + "/" + support["status"]] += 1
            else:
                stored_verdict_support["missing_or_ambiguous_source_join"] += 1
            point = vehicle.ground_point
            x, y = int(np.floor(point[0])), int(np.floor(point[1]))
            if any(point_in_polygon(point, p) for p in scene.get("footpath_polygons", [])):
                if not (0 <= x < width and 0 <= y < height) or masks["sidewalk"][y, x] == 0:
                    centre_mismatch += 1
            base = SceneEvidence(scene_id=scene["scene_id"], stationary_camera=frame.get("camera_stable") is True,
                                 confidence=scene.get("segmentation", {}).get("road", 0.),
                                 footpath_polygons=scene.get("footpath_polygons", []))
            resolved = gate.scene_for(vehicle, frame["timestamp"], base, video=report.get("kind") == "video",
                                      scene_revision=frame.get("scene_revision"))
            gate_status[resolved.source_evidence["footpath_contact_support"]["status"]] += 1
    unchanged = all(sha(run / name) == value for name, value in protected.items())
    if not unchanged:
        raise ValueError("A source artifact changed during audit")
    return {"run_id": run.name, "sources_sha256": protected, "frames": frames,
            "frames_without_recorded_camera_stability": missing_camera, "two_wheeler_contacts": contacts,
            "contact_spatial_status": dict(spatial), "contact_spatial_reasons": dict(reasons),
            "stored_footpath_verdict_by_spatial_support": dict(stored_verdict_support),
            "two_wheeler_centre_polygon_mask_disagreements": centre_mismatch,
            "context_status": dict(gate_status), "scenes": scene_receipts, "sources_unchanged": True,
            "source_decisions_recomputed": False, "fresh_inference": False,
            "provenance": {key: report[key] for key in ("synthetic", "provenance") if key in report} or {"status": "not_recorded"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(run.resolve()) for run in args.runs):
        raise ValueError("Output must be new and outside source runs")
    results = [audit_run(run) for run in args.runs]
    report = {"kind": "footpath_source_raster_audit", "completed": True, "results": results,
              "source_preservation_passed": all(r["sources_unchanged"] for r in results),
              "limits": ["Mask labels and polygons are model observations, not independent real traffic truth.",
                         "Grid mismatch is topology/approximation disagreement, not an observed violation error.",
                         "Missing camera stability is not inferred; no source legal decisions are changed."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    temporary.rename(output)
    print(json.dumps({"completed": True, "runs": len(results), "frames": sum(r["frames"] for r in results),
                      "contacts": sum(r["two_wheeler_contacts"] for r in results), "fresh_inference": False}))


if __name__ == "__main__":
    main()
