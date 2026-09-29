import json
import sys
import time
import zipfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ivp.jobs import job_lock
from ivp.media import load_image, png_bytes, write_json
from ivp.pipeline import Settings, process_frame
from ivp.worker import Cancelled, check_cancel, process_video, update

from .analyzer import Analyzer
from .render import annotate_detections


def process_run(run):
    request = json.loads((run / "request.json").read_text(encoding="utf-8"))
    try:
        check_cancel(run)
        analyzer = Analyzer(status=lambda text: update(run, "running", progress=0, message=text), cancel=lambda: check_cancel(run))
        settings = Settings(**request["settings"])
        saved_scene_revision = 0
        def finish():
            check_cancel(run)
            evidence = run / "evidence"
            if evidence.exists():
                with zipfile.ZipFile(run / "evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in evidence.glob("*.png"):
                        check_cancel(run)
                        archive.write(path, path.name)
            scene_folder = run / "scene_features"
            if scene_folder.exists():
                with zipfile.ZipFile(run / "scene_features.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                    for path in scene_folder.rglob("*"):
                        check_cancel(run)
                        if path.is_file():
                            archive.write(path, path.relative_to(scene_folder))
            summary = analyzer.summary()
            check_cancel(run)
            return summary

        with (run / "detections.jsonl").open("w", encoding="utf-8") as frames, (run / "events.jsonl").open("w", encoding="utf-8") as events:
            def hook(result, original, index, distance):
                nonlocal saved_scene_revision
                rgb, analysis, created = analyzer.analyze(result, original, video=request["info"]["kind"] == "video", scene_change=distance >= settings.scene_threshold)
                if getattr(analyzer, "scene_revision", 0) != saved_scene_revision:
                    saved_scene_revision = analyzer.scene_revision
                    scene_folder = run / "scene_features" / f"{saved_scene_revision:04d}"
                    scene_folder.mkdir(parents=True, exist_ok=True)
                    write_json(scene_folder / "scene.json", analyzer.scene_snapshot())
                    for name, mask in analyzer.scene_masks.items():
                        (scene_folder / f"{name}.png").write_bytes(png_bytes(mask))
                record = analysis.to_dict()
                if request["info"]["kind"] == "image":
                    (run / "detections.png").write_bytes(png_bytes(annotate_detections(result.color_rgb, analysis.detections)))
                record["frame"] = index
                record["source_size"] = list(result.source_size)
                record["processed_size"] = [result.color_rgb.shape[1], result.color_rgb.shape[0]]
                record["transform_to_source_pixel_centres"] = result.transform_to_source.tolist()
                record["scene_revision"] = saved_scene_revision
                record["signal_observations"] = getattr(analyzer, "signals", [])
                record["turn_observations"] = getattr(analyzer, "turn_observations", [])
                camera_stable = getattr(getattr(analyzer, "scene", None), "stationary_camera", None)
                if type(camera_stable) is bool:
                    # Record the actual gate used for this frame, not an
                    # inference from a later verdict or an archived scene.
                    record["camera_stable"] = camera_stable
                snapshot = getattr(getattr(analyzer, "events", None), "confirmation_snapshot", None)
                if callable(snapshot):
                    record["event_confirmation"] = deepcopy(snapshot())
                    indices = getattr(analyzer, "event_vehicle_indices", None)
                    if indices is not None:
                        if (not isinstance(indices, list) or
                                any(type(i) is not int or not 0 <= i < len(record["detections"]) for i in indices)):
                            raise ValueError("Event vehicle indices do not reference the recorded detections")
                        record["event_vehicle_indices"] = indices[:]
                        record["event_confirmation_indexing"] = {
                            "index_base": 0, "source_decision_indices": "/decisions",
                            "source_detection_index": "eligible list index -> /event_vehicle_indices -> /detections"}
                frames.write(json.dumps(record, allow_nan=False) + "\n")
                frames.flush()
                for event in created:
                    events.write(json.dumps(asdict(event)) + "\n")
                    evidence = run / "evidence"
                    evidence.mkdir(exist_ok=True)
                    (evidence / f"{event.event_id}.png").write_bytes(png_bytes(rgb))
                events.flush()
                return rgb

            if request["info"]["kind"] == "video":
                process_video(run, frame_hook=hook, final_hook=finish)
            else:
                original = load_image(request["source"])
                result = process_frame(original, settings, include_features=False, include_stages=False)
                rgb = hook(result, original, 0, 0)
                check_cancel(run)
                (run / "processed.png").write_bytes(png_bytes(rgb))
                write_json(run / "coordinates.json", {"source_size": result.source_size, "processed_size": [rgb.shape[1], rgb.shape[0]],
                           "transform_to_source_pixel_centres": result.transform_to_source.tolist()})
                write_json(run / "report.json", {"schema_version": 1, "kind": "image", "source": Path(request["source"]).name,
                           "settings": settings.to_dict(), "ai": finish()})
                check_cancel(run)
                update(run, "complete", progress=1, message="AI image analysis ready.")
    except Cancelled:
        for name in ("processed.png", "report.json"):
            (run / name).unlink(missing_ok=True)
        update(run, "cancelled", progress=0, message="AI processing cancelled. Original upload is unchanged.")
    except Exception as exc:
        update(run, "failed", progress=0, message=f"AI processing failed: {exc}. See worker.log; run Setup-AI.ps1 if dependencies are missing.")
        raise


def main():
    run = Path(sys.argv[1]).resolve()
    for attempt in range(20):
        try:
            with job_lock():
                process_run(run)
            return
        except RuntimeError as exc:
            if "Another video job" not in str(exc):
                raise
            time.sleep(.2)
    update(run, "failed", progress=0, message="Another processing job is active.")


if __name__ == "__main__":
    main()
