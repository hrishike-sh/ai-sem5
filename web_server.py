"""Loopback-only Traffic Lab API and static web application. No cloud dependencies."""
import argparse
import csv
import hashlib
import json
import math
import mimetypes
import re
import secrets
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ivp.downloads import FileServer
from ivp.inspection import inspect_region
from ivp.jobs import ROOT, cancel_job, history, new_run, now, refreshed_status, start_video
from ivp.media import load_image, png_bytes, preview_frame, probe_video, write_json, source_revision
from ivp.pipeline import PRESETS, Settings, histogram, process_frame
from ivp.workspace import run_details, source_name
from ivp.automatic import choose_settings
from ivp.stages import build_stages, manifest
from ivp.survey import survey
from ivp.evidence_reviews import EvidenceReviews
from ivp.library import RunLibrary
from ivp.scene_reviews import SceneReviews
from ivp.ai_assessments import saved_ai_page
from ivp.examples import examples, identify_example
from ivp.marking_policy import mask_export_manifest, review_only_policy

WEB = ROOT / "web"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"}
SAMPLE_NAMES = {"indian_riders.png": "Indian motorcycles", "indian_night.png": "Indian night traffic",
                'indian-motion.mp4': 'Indian traffic · crowded stress test', 'traffic-motion.avi': 'Pedestrian movement · OpenCV sample',
                'india-city.mp4': 'Indian city traffic - recommended automatic tracking demo',
                'india-delhi-overhead.mp4': 'Delhi junction - crowded autos, buses & bikes',
                'india-intersection.mp4': 'Delhi intersection - mixed street traffic', 'india-rickshaws.mp4': 'Delhi street - rickshaws & close occlusion',
                'highway-motion.mp4': 'Highway traffic · fixed camera demo',
                "street.jpg": "Street scene", "low_light.png": "Low light · simulated", "noisy.png": "Sensor noise · simulated",
                "crossing.jpg": "Zebra crossing", "sample_video.mp4": "Video · still image + test audio"}
EXAMPLES = examples(ROOT)
SAMPLE_NAMES.update({name: item['label'] for name, item in EXAMPLES.items()})


def sample_path(name):
    return EXAMPLES[name]['path'] if name in EXAMPLES else ROOT / 'demo' / name


DOWNLOADS = {"processed.png": "Processed image", "processed.mp4": "Processed video", "detections.png": "Detected objects",
             'helmet-observations.json':'Helmet observations and association evidence (JSON)',
             'helmet-observations.csv':'Helmet observations (CSV)', 'helmet-original.png':'Helmet check input frame',
             'motion-direction.jsonl':'Configured-direction samples (JSONL)',
             'motion-direction.json':'Configured-direction review summary (JSON)',
             'zebra-observations.json': 'Crossing occupancy observations (JSON)',
             'zebra-observations.csv': 'Crossing occupancy observations (CSV)',
             'zebra-original.png': 'Crossing check input frame',
             'motion-zebra.jsonl': 'Crossing occupancy samples (JSONL)',
             'motion-zebra.json': 'Crossing occupancy summary (JSON)',
             'motion-incidents.jsonl': 'Incident assessments (JSONL)', 'motion-incidents.csv': 'Incident assessments (CSV)',
             'motion-incidents.json': 'Incident review summary and configuration',
             'motion-tracks.png': 'Full-resolution tracks and speed', 'motion-flow.png': 'Local optical-flow arrows',
             'motion-evidence.zip': 'Motion analysis evidence package', 'motion-heatmap.png': 'Movement heatmap',
             'motion-mosaic.png': 'Peak activity comparison', 'motion-activity.csv': 'Motion timeline (CSV)',
             'motion-crossings.csv': 'Line crossings (CSV)', 'motion-tracks.csv': 'Track trajectories (CSV)',
             'motion-background.png': 'Estimated background', 'motion-raw-foreground.png': 'Raw foreground labels',
             'motion-shadows.png': 'Rejected shadow mask', 'motion-mask.png': 'Cleaned foreground mask',
             'survey-evidence.zip': 'Survey images and masks', 'fragments.csv': 'Fragment measurements (CSV)',
             'annotated-survey.png': 'Annotated marking survey',
             'processing-policy.json': 'Automatic processing policy',
             "ivp-stages.zip": "All IVP stages + explanations",
             "report.json": "Settings & measurements", "masks.zip": "Feature masks", "measurements.csv": "Frame measurements",
             "scene_changes.csv": "Scene changes", "coordinates.json": "Coordinate mapping", "events.jsonl": "AI events",
             "detections.jsonl": "AI frame decisions", "evidence.zip": "Event evidence", "scene_features.zip": "Scene features"}


def parse_settings(value):
    if not isinstance(value, dict) or set(value) - set(Settings.__dataclass_fields__):
        raise ValueError("Unknown processing settings.")
    defaults = Settings().to_dict()
    for name, item in value.items():
        expected = type(defaults[name])
        if expected in (int, float):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"{name} must be a finite number.")
            if expected is int and type(item) is not int:
                raise ValueError(f"{name} must be an integer.")
        elif type(item) is not expected:
            raise ValueError(f"Invalid type for {name}.")
    settings = Settings(**value)
    if settings.max_width not in (0,) and not 2 <= settings.max_width <= 8192:
        raise ValueError("Maximum width must be 0 or between 2 and 8192.")
    if settings.line_length > 8192:
        raise ValueError("Line length is too large.")
    return settings


def save_demo_assets(run, document, assets, binding):
    """Publish newly captured arrays in this run; never upgrade legacy runs."""
    folder = run / 'executed-ivp'
    folder.mkdir()
    document['source_binding'] = binding
    document['assets'] = {}
    for key, pixels in assets.items():
        if not re.fullmatch(r'[a-z0-9_-]{1,40}', key):
            raise ValueError('Invalid captured stage identifier.')
        encoded = png_bytes(pixels)
        (folder / (key + '.png')).write_bytes(encoded)
        document['assets'][key] = {'sha256': hashlib.sha256(encoded).hexdigest()}


def demo_view(run, report, files):
    document = report.get('ivp_demo')
    if (not isinstance(document, dict) or document.get('schema_version') != 1 or
            document.get('source_binding') != report.get('source_binding') or
            not report.get('source_binding')):
        return None
    assets, steps = document.get('assets'), document.get('steps')
    if not isinstance(assets, dict) or not 1 <= len(assets) <= 40 or not isinstance(steps, list) or len(steps) > 32:
        return None
    urls = {}
    for key, details in assets.items():
        if not isinstance(key, str) or not re.fullmatch(r'[a-z0-9_-]{1,40}', key) or not isinstance(details, dict):
            return None
        path = run / 'executed-ivp' / (key + '.png')
        if not path.resolve().is_relative_to(run.resolve()) or not path.is_file():
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != details.get('sha256'):
            return None
        urls[key] = files.url(path)
    if any(not isinstance(s, dict) or not isinstance(s.get('before'), str) or not isinstance(s.get('after'), str)
           or s['before'] not in urls or s['after'] not in urls for s in steps):
        return None
    if not all(isinstance(document.get(key), str) and document[key] in urls for key in ('original', 'final')):
        return None
    return {**document, 'urls': urls,
            'result_download': files.url(run / 'executed-ivp' / (document['final'] + '.png'), True)}


class Lab:
    def __init__(self):
        self.files = FileServer()
        self.media = {}
        self.lock = threading.Lock()
        self.cache = None
        self.auto_profiles = {}
        self.library = RunLibrary(ROOT/'runs')

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.evidence_reviews = EvidenceReviews(ROOT, self.executor, self.files)
        self.scene_reviews = SceneReviews(ROOT, self.executor)
        self.token = secrets.token_urlsafe(32)

    def close(self):
        self.scene_reviews.close()
        self.files.close()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def register(self, path):
        path = Path(path)
        revision = source_revision(path)
        info = probe_video(path) if path.suffix.lower() in (".mp4", ".mov", ".avi") else None
        if info is None:
            rgb = load_image(path)
            info = {"kind": "image", "width": rgb.shape[1], "height": rgb.shape[0]}
        key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
        info['source_revision'] = revision
        self._check_source(path, info)
        example = identify_example(path, EXAMPLES)
        if example:
            info['example'] = example
        self._check_source(path, info)
        registry = ROOT / "temp" / "webmedia"
        registry.mkdir(parents=True, exist_ok=True)
        write_json(registry / (key + ".json"), {"path": str(path.resolve())})
        previous = self.media.get(key)
        if previous is None or previous[1].get('source_revision') != info['source_revision']:
            self.auto_profiles.pop(key, None)
        self.media[key] = (path, info)
        self.cache = None
        return {"id": key, "name": path.name, "size_bytes": path.stat().st_size, **info,
                **({'original_video': self.files.url(path)} if info['kind'] == 'video' and example and example.get('local_source') else {})}

    def resume(self, key):
        if not isinstance(key, str) or len(key) != 24 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError("Previous footage is unavailable; choose a file again.")
        record = ROOT / "temp" / "webmedia" / (key + ".json")
        try:
            path = Path(json.loads(record.read_text())["path"]).resolve()
            explicit_example = any(path == item['path'].resolve() for item in EXAMPLES.values())
            if not explicit_example and not any(path.is_relative_to((ROOT / folder).resolve()) for folder in ("uploads", "demo")):
                raise ValueError("Saved source is outside the media folders.")
            if hashlib.sha256(str(path).encode()).hexdigest()[:24] != key:
                raise ValueError("Saved source identity changed.")
            return self.register(path)
        except (OSError, KeyError, TypeError) as exc:
            raise ValueError("Previous footage is unavailable; choose a file again.") from exc

    @staticmethod
    def _check_source(path, info):
        # Real registrations always record a revision. Older direct frame
        # providers may omit it; do not invent a content-digest requirement.
        expected = info.get('source_revision')
        if expected is None:
            return
        try:
            current = source_revision(path)
        except OSError as exc:
            raise ValueError('Source is no longer available. Reopen the footage before continuing.') from exc
        if current != expected:
            raise ValueError('The source changed since it was opened. Reopen the footage before continuing.')

    def _source(self, data):
        if data.get("media") not in self.media:
            raise ValueError("Choose a sample or upload your footage again.")
        path, info = self.media[data["media"]]
        requested = data.get('source_revision')
        if requested is not None and requested != info.get('source_revision'):
            raise ValueError('The source changed since it was opened. Reopen the footage before continuing.')
        self._check_source(path, info)
        return path, info

    def frame(self, data):
        path, info = self._source(data)
        stamp = data.get("timestamp", 0)
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("Invalid frame timestamp.")
        if info["kind"] == "video":
            stamp = min(stamp, max(0, info["duration"] - .001))
            rgb, stamp = preview_frame(path, info, stamp)
        else:
            rgb, stamp = load_image(path), 0.
        self._check_source(path, info)
        return path, info, rgb, stamp

    def processed(self, data):
        path, info = self._source(data)
        settings, _ = self.resolve_settings(data)
        self._check_source(path, info)
        key = (data.get("media"), data.get("timestamp", 0), settings)
        if self.cache and self.cache[0] == key:
            return self.cache[1]
        path, info, rgb, stamp = self.frame(data)
        result = process_frame(rgb, settings, stamp)
        self._check_source(path, info)
        value = path, info, result, settings
        self.cache = key, value
        return value

    def resolve_settings(self, data):
        mode = data.get('mode', 'manual')
        if mode not in ('automatic', 'manual'):
            raise ValueError('Unknown processing mode.')
        if mode == 'manual':
            return parse_settings(data.get('settings', {})), {'mode': 'manual'}
        key = data.get('media')
        if key not in self.media:
            raise ValueError('Choose footage first.')
        path, info = self._source(data)
        if key not in self.auto_profiles:
            _, info, rgb, stamp = self.frame({**data, 'timestamp': 0})
            settings, explanation = choose_settings(rgb)
            self._check_source(path, info)
            self.auto_profiles[key] = settings, {**explanation, 'mode': 'automatic',
                'reference_timestamp': stamp,
                'scope': 'Fixed settings from the first frame for the whole video.' if info['kind'] == 'video' else 'Settings selected from this image.'}
        return self.auto_profiles[key]

    def preview(self, data):
        path, info, result, settings = self.processed(data)
        folder = ROOT / "temp" / "webpreviews" / data["media"]
        folder.mkdir(parents=True, exist_ok=True)
        version = secrets.token_hex(6)
        def asset(name, pixels):
            target = folder / (name + ".png")
            target.write_bytes(png_bytes(pixels))
            return self.files.url(target) + "?v=" + version
        before = result.stages["Prepared"]
        graph = build_stages(result, settings)
        steps = manifest(graph)
        for step, entry in zip(steps, graph):
            step['url'] = asset('ivp-' + step['id'], entry['image'])
            step['download'] = self.files.url(folder / ('ivp-' + step['id'] + '.png'), True)
        return {"name": path.name, "timestamp": result.timestamp, "settings": settings.to_dict(),
                "source_size": result.source_size, "size": [result.color_rgb.shape[1], result.color_rgb.shape[0]],
                "original": asset("original", before), "processed": asset("processed", result.color_rgb),
                "features": {name: asset(f"feature-{i}", pixels) for i, (name, pixels) in enumerate(result.features.items())},
                "stages": {name: asset(f"stage-{i}", pixels) for i, (name, pixels) in enumerate(result.stages.items())},
                "measurements": result.measurements, "candidates": result.candidates,
                "histograms": {"original": histogram(before), "processed": histogram(result.color_rgb)},
                "mapping": result.transform_to_source.tolist(), 'ivp_steps': steps,
                'automatic': self.resolve_settings(data)[1]}

    def marking_survey(self, data):
        # Always measure original colour at a fixed resolution; enhancement
        # sliders must not manufacture paint coverage or change its geometry.
        path, info, rgb, stamp = self.frame(data)
        revision = info['source_revision']
        if revision != source_revision(path):
            raise ValueError('The source changed since it was opened. Reopen it before measuring markings.')
        settings = Settings(max_width=1280)
        prepared = process_frame(rgb, settings, stamp)
        capture = data.get('capture_presentation', False)
        if type(capture) is not bool:
            raise ValueError('Invalid stage capture option.')
        report, images = survey(prepared.stages['Prepared'], data.get('rectangle'), prepared.transform_to_source,
                                **({'capture_stages': True} if capture else {}))
        report.update({'source': path.name, 'kind': 'ivp_marking_survey', 'timestamp_seconds': stamp,
                       'created_at': now(), 'source_size': prepared.source_size,
                       'source_binding': {'media_id': data['media'], 'source_revision': revision,
                           'timestamp_seconds': stamp, 'coordinate_space': 'display-oriented analysis pixel centres',
                           'transform_to_source_pixel_centres': prepared.transform_to_source.tolist()},
                       'settings': settings.to_dict()})
        if revision != source_revision(path):
            raise ValueError('The source changed during the survey. Reopen it and try again.')
        run = new_run()
        if capture:
            save_demo_assets(run, report['ivp_demo'],
                             {key: images['executed-' + key] for key in report['ivp_demo']['asset_keys']}, report['source_binding'])
        write_json(run / 'report.json', report)
        (run / 'annotated-survey.png').write_bytes(png_bytes(images['annotated-survey']))
        fields = ['id', 'status', 'semantic_class', 'eligible_for_geometry', 'eligible_for_rules', 'reason',
                  'area_px', 'bbox_x', 'bbox_y', 'bbox_width_px', 'bbox_height_px',
                  'long_side_px', 'short_side_px', 'orientation_deg_clockwise_from_right',
                  'touches_roi_boundary', 'source_centre_x', 'source_centre_y']
        with (run / 'fragments.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report['fragments'])
        with zipfile.ZipFile(run / 'survey-evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, pixels in images.items():
                archive.writestr(name + '.png', png_bytes(pixels))
            archive.write(run / 'report.json', 'report.json')
            archive.write(run / 'fragments.csv', 'fragments.csv')
        write_json(run / 'status.json', {'state': 'complete', 'progress': 1,
                   'message': 'IVP marking survey ready.', 'updated_at': now()})
        return self.run(run.name)

    def export(self, data):
        goal = data.get('ivp_goal')
        if goal is not None and goal not in ('colour', 'regions', 'edges'):
            raise ValueError('Choose a colour image, region map or edge map.')
        path, info, result, settings = self.processed(data)
        policy = self.resolve_settings(data)[1]
        masks = data.get("masks", [])
        if not isinstance(masks, list) or any(name not in result.features for name in masks):
            raise ValueError("Unknown feature mask.")
        self._check_source(path, info)
        binding = {'media_id': data['media'], 'source_name': path.name,
                   'source_revision': info.get('source_revision') or source_revision(path),
                   'timestamp_seconds': result.timestamp,
                   'transform_to_source_pixel_centres': result.transform_to_source.tolist()}
        run = new_run()
        (run / "processed.png").write_bytes(png_bytes(result.color_rgb))
        with zipfile.ZipFile(run / "masks.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in masks:
                archive.writestr(name.replace(" ", "_") + ".png", png_bytes(result.features[name]))
            archive.writestr('manifest.json', json.dumps(mask_export_manifest(masks, binding), indent=2))
        graph = build_stages(result, settings)
        demonstration = None
        if goal is not None:
            from ivp.presentation import image_presentation
            demonstration, captured = image_presentation(result, settings, graph, goal)
            save_demo_assets(run, demonstration, captured, binding)
        with zipfile.ZipFile(run / 'ivp-stages.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for i, step in enumerate(graph):
                archive.writestr(f"{i+1:02d}-{step['id']}.png", png_bytes(step['image']))
            archive.writestr('stages.json', json.dumps(manifest(graph), indent=2))
            if demonstration:
                archive.writestr('executed-steps.json', json.dumps(demonstration, indent=2))
                for key in demonstration['asset_keys']:
                    archive.write(run / 'executed-ivp' / (key + '.png'), 'executed/' + key + '.png')
            archive.writestr('processing.json', json.dumps({'source': path.name,
                'source_binding': binding,
                'timestamp': result.timestamp, 'settings': settings.to_dict(),
                'policy': policy,
                'marking_policy': review_only_policy(),
                'transform_to_source_pixel_centres': result.transform_to_source.tolist()}, indent=2))
        write_json(run / "report.json", {"schema_version": 1, "source": path.name,
                   'source_binding': binding,
                   "kind": "video_frame" if info["kind"] == "video" else "image", "created_at": now(),
                   "timestamp_seconds": result.timestamp, "settings": settings.to_dict(),
                   "measurements": result.measurements, "candidates": result.candidates,
                   **({'ivp_demo': demonstration} if demonstration else {}),
                   'ivp_stages': manifest(graph), 'processing_policy': policy,
                   "source_size": result.source_size, "processed_size": [result.color_rgb.shape[1], result.color_rgb.shape[0]],
                   "transform_to_source_pixel_centres": result.transform_to_source.tolist(), "selected_masks": masks})
        try:
            self._check_source(path, info)
        except ValueError as exc:
            write_json(run / 'status.json', {'state': 'failed', 'progress': 0,
                       'message': str(exc), 'updated_at': now()})
            raise
        write_json(run / "status.json", {"state": "complete", "progress": 1, "message": "Image export ready.", "updated_at": now()})
        return self.run(run.name)

    def run(self, name):
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Invalid run identifier.")
        run = ROOT / "runs" / name
        if not (run / "status.json").is_file():
            raise ValueError("Saved run was not found.")
        details = run_details(run)
        links = [{"name": file, "label": label, "url": self.files.url(run / file, True)} for file, label in DOWNLOADS.items() if (run / file).is_file()]
        status = refreshed_status(run)
        if status.get("state") != "complete":
            links = []
            # New neutral-observation summaries are authoritative only after a
            # complete source-checked run, including legacy partial reports.
            details={k:v for k,v in details.items() if k not in {'helmet','zebra'}}
            if isinstance(details.get('motion'),dict):
                details['motion']={k:v for k,v in details['motion'].items() if k not in {'zebra','direction'}}
        preview = None
        if status.get("state") == "complete":
            incident_summary = details.get('motion',{}).get('incidents',{})
            for event in incident_summary.get('events',[]):
                thumbnail = event.get('thumbnail')
                if (isinstance(thumbnail,str) and re.fullmatch(r'incident-\d{6}\.png',thumbnail)
                        and (run/thumbnail).is_file()):
                    event['thumbnail_url'] = self.files.url(run/thumbnail)
            for file in ("processed.mp4", "detections.png", "processed.png", "annotated-survey.png"):
                if (run / file).exists():
                    preview = {"url": self.files.url(run / file), "video": file.endswith(".mp4"), "name": file}
                    break
        return {"id": name, "source": source_name(details), "status": status, "downloads": links, "preview": preview, "report": details,
                **({'demonstration': demo_view(run, details, self.files)} if status.get('state') == 'complete' and 'ivp_demo' in details else {})}

    def post(self, route, data):
        if route == '/api/helmet':
            path,info=self._source(data)
            stamp=data.get('timestamp',0)
            if (isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
                    or stamp<0 or info['kind']=='image' and stamp!=0
                    or info['kind']=='video' and stamp>=info['duration']):
                raise ValueError('Choose a valid helmet preview position.')
            return self.run(start_video(path,info,Settings(max_width=1280),[],
                           helmet={'timestamp':stamp,'media_id':data['media']}).name)
        if route == '/api/zebra':
            from ivp.zebra import validate_region
            path, info = self._source(data)
            stamp = data.get('timestamp',0)
            if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
                raise ValueError('Choose a finite crossing preview position.')
            region = validate_region(data.get('region'),data['media'],info['source_revision'],stamp)
            if info['kind']=='image' and stamp != 0:
                raise ValueError('An image crossing review must use timestamp zero.')
            if info['kind']=='video' and stamp >= info['duration']:
                raise ValueError('Crossing frame is outside this video.')
            return self.run(start_video(path,info,Settings(max_width=1280),[],
                           zebra={'region':{k:v for k,v in region.items() if k!='region_id'},'timestamp':stamp}).name)
        if route == '/api/evidence-review':
            if self.scene_reviews.active:
                raise RuntimeError('A scene review is already running. Wait for it to finish before creating an evidence review.')
            return self.evidence_reviews.start(data.get('source_run'),data.get('use_model',False))
        if route == '/api/motion':
            from ivp.motion import MotionAnalyzer
            if data.get('media') not in self.media:
                raise ValueError('Choose a traffic video first.')
            path, info = self._source(data)
            if info['kind'] != 'video':
                raise ValueError('Motion analysis needs a video. Choose the Indian traffic moving-video sample.')
            if info.get('source_revision') != source_revision(path):
                raise ValueError('This video changed since it was opened. Reopen it and mark the road calibration again.')
            calibration = data.get('calibration')
            if calibration is not None:
                if (not isinstance(calibration, dict) or
                        calibration.get('media_id') != data['media'] or
                        calibration.get('source_revision') != info['source_revision'] or
                        calibration.get('reference_frame') != 0):
                    raise ValueError('Calibration belongs to different or outdated footage. Mark four road corners on this video first.')
            config = {'axis': data.get('axis','horizontal'), 'position': data.get('position',.65),
                      'rectangle': data.get('rectangle'), 'calibration': data.get('calibration'),
                      'target_rectangle':data.get('target_rectangle'), 'tracking_mode':data.get('tracking_mode','ivp'),
                      'output_view':data.get('output_view','tracks')}
            if 'speed_display' in data:
                from ivp.speed_display import validate_speed_display
                config['speed_display'] = validate_speed_display(data['speed_display'], calibration)
            MotionAnalyzer(**config)
            if data.get('direction') is not None:
                from ivp.configured_direction import validate_config
                if config['tracking_mode']!='vehicles':
                    raise ValueError('Configured-direction review needs automatic vehicle tracking.')
                config['direction']=validate_config(data['direction'],data['media'],info['source_revision'])
                if config['rectangle'] is None:
                    pts=config['direction']['points']
                    config['rectangle']=[max(0,min(p[0] for p in pts)-.03),max(0,min(p[1] for p in pts)-.03),
                                         min(1,max(p[0] for p in pts)+.03),min(1,max(p[1] for p in pts)+.03)]
                    index=1 if config['axis']=='horizontal' else 0
                    config['position']=min(.9,max(.1,(config['rectangle'][index]+config['rectangle'][index+2])/2))
            if data.get('zebra') is not None:
                from ivp.zebra import validate_region
                if config['tracking_mode'] != 'vehicles':
                    raise ValueError('Crossing occupancy needs automatic vehicle tracking.')
                config['zebra'] = validate_region(data['zebra'],data['media'],info['source_revision'],0.)
                if config['rectangle'] is None:
                    # Focus neutral occupancy tracking around the reviewed band,
                    # leaving surrounding texture for first-frame registration.
                    points=config['zebra']['points']
                    bounds=[min(p[0] for p in points),min(p[1] for p in points),
                            max(p[0] for p in points),max(p[1] for p in points)]
                    config['rectangle']=[max(0,bounds[0]-.12),max(0,bounds[1]-.12),
                                         min(1,bounds[2]+.12),min(1,bounds[3]+.12)]
                    index=1 if config['axis']=='horizontal' else 0
                    config['position']=min(.9,max(.1,(config['rectangle'][index]+config['rectangle'][index+2])/2))
            from ivp.incidents import validate_config
            incidents = validate_config(data.get('incidents'))
            if incidents and incidents['enabled']:
                if info.get('example', {}).get('synthetic') is True:
                    incidents['synthetic'] = True
                if config['tracking_mode'] != 'vehicles':
                    raise ValueError('Incident review needs automatic vehicle tracking. Select Vehicles · local detector + IVP.')
                config['incidents'] = incidents
            self._check_source(path, info)
            from ivp.feature_scope import validate_feature_scope
            feature=data.get('ui_feature')
            validate_feature_scope(feature,config)
            run_info={**info,**({'ui_feature':feature} if feature is not None else {})}
            return self.run(start_video(path,run_info,Settings(max_width=960),[],motion=config).name)
        if route == '/api/survey':
            return self.marking_survey(data)
        if route == "/api/validate-settings":
            return parse_settings(data.get("settings", {})).to_dict()
        if route == "/api/sample":
            name = data.get("name")
            if name not in SAMPLE_NAMES or not sample_path(name).is_file():
                raise ValueError("Sample not found.")
            return self.register(sample_path(name))
        if route == "/api/preview":
            return self.preview(data)
        if route == "/api/region":
            return inspect_region(self.processed(data)[2], data.get("rectangle"))
        if route == "/api/export":
            return self.export(data)
        if route == "/api/job":
            # Validate without running expensive IVP before starting the worker.
            if data.get("media") not in self.media:
                raise ValueError("Choose footage first.")
            path, info = self._source(data)
            ai = data.get("ai", False)
            if type(ai) is not bool or (not ai and info["kind"] != "video"):
                raise ValueError("Use image export for still images.")
            settings, policy = self.resolve_settings(data)
            masks = data.get("masks", [])
            if not isinstance(masks, list) or any(not isinstance(v, str) for v in masks):
                raise ValueError("Invalid masks.")
            allowed_masks = {"Grayscale", "Hue", "Saturation", "Value", "Colour mask", "White marking mask", "Edges", "Threshold", "Morphology", "Contours", "Lines", "Road candidates", "Stripe mask", "Sobel magnitude", "Laplacian", "Morphological gradient", "Top hat", "Black hat"}
            if any(v not in allowed_masks for v in masks):
                raise ValueError("Unknown feature mask.")
            self._check_source(path, info)
            run = start_video(path, info, settings, masks, ai=ai)
            write_json(run / 'processing-policy.json', policy)
            return self.run(run.name)
        if route == "/api/cancel":
            run = self.run(data.get("id"))
            if run["status"]["state"] in ("queued", "running", "encoding"):
                cancel_job(ROOT / "runs" / run["id"])
            return {"message": "Cancellation requested."}
        if route == "/api/review":
            if self.evidence_reviews.future and not self.evidence_reviews.future.done():
                raise RuntimeError('An evidence review is already running. Wait for it to finish before starting a scene overview.')
            if self.scene_reviews.active:
                raise RuntimeError('A scene overview is already running. Cancel it or wait for it to finish.')
            path, info, rgb, stamp = self.frame(data)
            if info['source_revision'] != source_revision(path):
                raise ValueError('The source changed since it was opened. Reopen it before requesting an overview.')
            return self.scene_reviews.start(rgb, stamp, data['media'], info['source_revision'])
        if route == '/api/review/cancel':
            return self.scene_reviews.cancel(data.get('id'))
        raise ValueError("Unknown operation.")


def create_server(port=8521):
    lab = Lab()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def permitted(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed:
                self.respond({'error': 'Localhost access only.'}, 403)
                return False
            return True

        def respond(self, value, status=200):
            if status >= 400 and self.command == 'POST' and not getattr(self, 'body_started', False):
                # On Windows, closing with unread request bytes can reset the
                # connection before the client receives this actionable error.
                # Drain only small rejected bodies, with a short bounded wait.
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    length = 0
                if 0 < length <= 65536:
                    previous_timeout = self.connection.gettimeout()
                    try:
                        self.connection.settimeout(.25)
                        self.rfile.read(length)
                    except OSError:
                        pass
                    finally:
                        self.connection.settimeout(previous_timeout)
                self.body_started = True
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.permitted():
                return
            parsed = urlparse(self.path)
            route, query = parsed.path, parse_qs(parsed.query)
            try:
                if route == "/api/health":
                    return self.respond({"app": "ai-sem5-web", "version": 1})
                if route == "/api/config":
                    from traffic_ai.runtime import runtime_root, python_path
                    from traffic_ai.local_models import helmet_availability
                    vehicle_ready=python_path().is_file() and (runtime_root()/"models"/"yolo11s.pt").is_file()
                    return self.respond({"speed_display_v1":True,"vehicle_tracking_ready":vehicle_ready,"helmet":helmet_availability(),"defaults": Settings().to_dict(), "presets": {k: v.to_dict() for k, v in PRESETS.items()},
                                         "samples": [{"name": k, "label": v} for k, v in SAMPLE_NAMES.items() if sample_path(k).is_file()]})
                if route == "/api/media":
                    with lab.lock:
                        return self.respond(lab.resume(query.get("id", [None])[0]))
                if route == "/api/runs":
                    return self.respond([lab.run(p.parent.name) for p in history()])
                if route == '/api/run-library':
                    return self.respond(lab.library.page(query.get('q',[''])[0],query.get('status',[''])[0],
                                                        query.get('before',[''])[0],int(query.get('limit',['30'])[0])))
                if route == "/api/run":
                    return self.respond(lab.run(query.get("id", [None])[0]))
                if route == '/api/ai-assessments':
                    return self.respond(saved_ai_page(ROOT, query.get('id', [None])[0],
                                                      query.get('cursor', [None])[0],
                                                      query.get('verdict', ['cannot_assess'])[0],
                                                      query.get('rule', ['all'])[0]))
                if route == '/api/evidence-review':
                    return self.respond(lab.evidence_reviews.get(query.get('id',[None])[0],query.get('source',[None])[0],
                                                               query.get('cursor',[None])[0],query.get('status',['all'])[0]))
                if route == "/api/review":
                    return self.respond(lab.scene_reviews.get(query.get('id', [None])[0]))
                static = {"/": "index.html", "/ai": "index.html", "/ai/": "index.html", "/app.js": "app.js", "/ivp_features.js":"ivp_features.js", "/styles.css": "styles.css"}
                if route not in static:
                    return self.respond({"error": "Not found"}, 404)
                path = WEB / static[route]
                body = path.read_bytes().replace(b"__REQUEST_TOKEN__", lab.token.encode())
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' http://127.0.0.1:*; media-src http://127.0.0.1:*; connect-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.permitted():
                return
            if self.headers.get("X-Traffic-Token") != lab.token:
                return self.respond({"error": "Reload the app to refresh your session."}, 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                return self.respond({"error": "Cross-origin request rejected."}, 403)
            if not lab.lock.acquire(blocking=False):
                return self.respond({"error": "Another operation is running. Try again shortly."}, 409)
            try:
                self.connection.settimeout(120)
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length", "0"))
                limit = 1024 ** 3 if parsed.path == "/api/upload" else 65536
                if not 0 < length <= limit:
                    raise ValueError("File is empty or exceeds the 1 GB upload limit." if limit > 65536 else "Invalid request size.")
                if parsed.path == "/api/upload":
                    name = parse_qs(parsed.query).get("name", [""])[0].replace("\\", "/").rsplit("/", 1)[-1]
                    if Path(name).suffix.lower() not in EXTENSIONS or len(name) > 180 or any(c in name for c in '<>:"|?*\x00'):
                        raise ValueError("Choose a JPG, PNG, MP4, MOV or AVI file with a valid filename.")
                    folder = ROOT / "uploads" / secrets.token_hex(12)
                    folder.mkdir(parents=True)
                    path = folder / name
                    self.body_started = True
                    try:
                        with path.open("wb") as output:
                            remaining = length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("Upload was interrupted; select the file again.")
                                output.write(chunk)
                                remaining -= len(chunk)
                        result = lab.register(path)
                    except Exception:
                        # Only this request's newly created, rejected copy.
                        # Accepted footage and the user's original are untouched.
                        try:
                            path.unlink(missing_ok=True)
                            folder.rmdir()
                        except OSError:
                            import traceback
                            traceback.print_exc()
                        raise
                else:
                    self.body_started = True
                    data = json.loads(self.rfile.read(length))
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object.")
                    result = lab.post(parsed.path, data)
                self.respond(result)
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception:
                import traceback
                traceback.print_exc()
                self.respond({"error": "Processing failed. See temp/server.log for details."}, 500)
            finally:
                lab.lock.release()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.lab = lab
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8521)
    args = parser.parse_args()
    server = create_server(args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.lab.close()
