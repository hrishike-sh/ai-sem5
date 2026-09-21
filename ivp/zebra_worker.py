"""Explicit current-frame vehicle/band check, using only the installed detector."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from .jobs import job_lock
from .media import load_image, png_bytes, preview_frame, write_json
from .pipeline import Settings, process_frame
from .worker import Cancelled, check_cancel, check_source, update
from .zebra import NOTICE, POLICY, assess, draw, validate_region


def process_run(run):
    request = json.loads((run/'request.json').read_text(encoding='utf-8'))
    source, info = request['source'], request['info']
    try:
        check_cancel(run)
        revision = check_source(source, info.get('source_revision'))
        config = request['zebra']
        region = validate_region(config['region'], config['region']['media_id'], revision, config['timestamp'])
        from traffic_ai.runtime import runtime_root
        model_path = runtime_root()/'models'/'yolo11s.pt'
        if not model_path.is_file():
            raise RuntimeError(f'Local vehicle model weights are missing: {model_path}. Restore the project’s expected local yolo11s.pt weights, then recheck the detector. Setup-AI.ps1 installs dependencies, not these weights.')
        update(run, 'running', progress=.05, message='Checking the reviewed crossing band with the installed vehicle detector.')
        if info['kind'] == 'video':
            original, stamp = preview_frame(source, info, config['timestamp'])
        else:
            original, stamp = load_image(source), 0.
        if abs(stamp-config['timestamp']) > 1e-6:
            raise ValueError('Decoded frame differs from the reviewed frame. Reopen the preview and review the band again.')
        check_cancel(run)
        check_source(source, revision)
        result = process_frame(original, Settings(**request['settings']), timestamp=stamp,
                               include_features=False, include_stages=False)
        from traffic_ai.perception import Perception
        from traffic_ai.vehicle_motion import ambiguous_overlaps
        detector = Perception(helmet=False)
        detections = [d for d in detector.detect(result.color_rgb) if d.label in {'bicycle','car','motorcycle','bus','truck'}]
        check_cancel(run)
        check_source(source, revision)
        ambiguous = ambiguous_overlaps(np.asarray([d.box for d in detections]))
        rows = [assess(region,d.box,result.color_rgb.shape,label=d.label,confidence=d.confidence,
                       track_id=d.track_id,timestamp=stamp,
                       uncertain='overlapping vehicle hypotheses' if i in ambiguous else None)
                for i,d in enumerate(detections)]
        binding = {'media_id': region['media_id'], 'source_revision': revision, 'timestamp_seconds': stamp,
                   'analysis_size': [result.color_rgb.shape[1],result.color_rgb.shape[0]],
                   'transform_to_source_pixel_centres': result.transform_to_source.tolist()}
        summary = {'schema_version':1,'notice':NOTICE,'reviewed_region':region,
                   'source_binding':binding,'observations':rows,'detector':detector.manifest,
                   'synthetic': info.get('example',{}).get('synthetic') is True,
                   'counts':{status:sum(r['status']==status for r in rows)
                             for status in ('observed','no_overlap','withheld')}, **POLICY}
        from .input_preparation import preparation_record
        summary['input_preparation']=preparation_record(original,result,Settings(**request['settings']),binding)
        (run/'zebra-original.png').write_bytes(png_bytes(result.color_rgb))
        (run/'processed.png').write_bytes(png_bytes(draw(result.color_rgb,region,rows)))
        write_json(run/'zebra-observations.json',summary)
        with (run/'zebra-observations.csv').open('w',newline='',encoding='utf-8') as stream:
            fields = ['timestamp_seconds','track_id','status','label','reason','overlap_area_px2',
                      'vehicle_overlap_fraction','crossing_overlap_fraction','basis','legal_violation']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
        write_json(run/'coordinates.json',binding)
        check_cancel(run)
        check_source(source,revision)
        write_json(run/'report.json',{'schema_version':1,'kind':'zebra_occupancy_frame',
                   'source':Path(source).name,'source_info':info,'zebra':summary})
        check_cancel(run)
        check_source(source,revision)
        update(run,'complete',progress=1,message='Crossing occupancy check ready. Image-space observation only.')
    except Cancelled:
        update(run,'cancelled',progress=0,message='Crossing check cancelled. Original footage is unchanged.')
    except Exception as exc:
        update(run,'failed',progress=0,message=f'Crossing check failed: {exc}')
        raise


def main():
    run=Path(sys.argv[1]).resolve()
    for attempt in range(20):
        try:
            with job_lock():
                process_run(run)
            return
        except RuntimeError as exc:
            if 'Another video job' not in str(exc):
                raise
            time.sleep(.2)
    update(run,'failed',progress=0,message='Another processing worker is active.')


if __name__ == '__main__':
    main()
