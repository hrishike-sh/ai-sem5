"""Incremental motion analysis using the shared timed-video encoder and job lock."""
import csv
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import cv2

from .jobs import job_lock
from .media import png_bytes, write_json
from .motion import MotionAnalyzer
from .incidents import IncidentAnalyzer
from .incident_export import IncidentEvidence, draw_geometry
from .performance import configure_opencv_threads
from .worker import check_cancel, process_video, update
from .zebra import Occupancy, draw as draw_zebra, validate_region


def run_motion(run):
    # The shared job lock keeps this CPU budget within one active video worker.
    runtime = configure_opencv_threads()
    cv2.setRNGSeed(0)
    request = json.loads((run/'request.json').read_text(encoding='utf-8'))
    fps=request['info'].get('fps') or 30
    motion_config = dict(request['motion'])
    incident_config = motion_config.pop('incidents', None)
    direction_config = motion_config.pop('direction',None)
    if direction_config:
        from .configured_direction import validate_config,DirectionReview
        direction_config=validate_config({k:v for k,v in direction_config.items() if k!='config_id'},
            direction_config['media_id'],request['info']['source_revision'])
        direction_review=DirectionReview(direction_config)
    else:direction_review=None
    zebra_region = motion_config.pop('zebra', None)
    if zebra_region:
        zebra_region = validate_region({k:v for k,v in zebra_region.items() if k!='region_id'},
            zebra_region['media_id'],request['info']['source_revision'],0.)
    zebra = Occupancy(zebra_region) if zebra_region else None
    provenance = request['info'].get('example')
    source_synthetic = bool(provenance and provenance.get('synthetic') is True)
    if source_synthetic and incident_config:
        incident_config = {**incident_config, 'synthetic': True}
    engine = MotionAnalyzer(**{**motion_config,'frame_interval':1/fps})
    incidents = IncidentAnalyzer(incident_config, motion_config.get('calibration'), 1/fps)
    with IncidentEvidence(run, incidents) as evidence, \
         (run/'motion-activity.csv').open('w', newline='', encoding='utf-8') as activity_file, \
         (run/'motion-crossings.csv').open('w', newline='', encoding='utf-8') as crossing_file, \
         (run/'motion-tracks.csv').open('w', newline='', encoding='utf-8') as track_file:
        activity = csv.DictWriter(activity_file, fieldnames=['timestamp_seconds','state','moving_area_percent','active_tracks','crossings_positive','crossings_negative','global_flow_px','unresolved_large_regions','flow_speed_px_s','flow_points','speed_estimates','stabilization_x_px','stabilization_y_px']+[f'camera_m{i}' for i in range(6)])
        activity.writeheader()
        crossings = csv.DictWriter(crossing_file, fieldnames=['timestamp_seconds','track_id','direction','x','y'])
        crossings.writeheader()
        tracks = csv.writer(track_file)
        display_columns = ['speed_display_drawn','speed_display_reason'] if getattr(engine,'speed_display',None) else []
        tracks.writerow(['timestamp_seconds','track_id','centre_x_px','centre_y_px','bbox_x','bbox_y','bbox_width','bbox_height','speed','speed_unit','speed_status','speed_window_seconds','model_label','detection_confidence','feature_count','tracking_state']+display_columns)
        def frame(result, original, number, distance):
            output, metrics, events, visible = engine.step(result.color_rgb, result.timestamp, distance >= .45)
            if direction_review:
                from .configured_direction import draw as draw_direction
                rows=direction_review.step(visible,result.timestamp,result.color_rgb.shape[:2],
                    registration_valid=not engine.stabilizer.failed and engine.registration_observed,
                    scene_changed=distance>=.45 or metrics['state']=='camera/scene change',valid_mask=engine.stabilizer.valid_mask)
                affine=np.vstack([engine.stabilizer.matrix,[0,0,1]])
                binding={'source_revision':request['info']['source_revision'],'frame':number,
                         'coordinate_space':'registered first-frame analysis image extent',
                         'current_frame_to_source_pixel_centres':(result.transform_to_source @ np.linalg.inv(affine)).tolist()}
                with (run/'motion-direction.jsonl').open('a',encoding='utf-8') as stream:
                    for row in rows:stream.write(json.dumps({**row,'source_binding':binding},allow_nan=False)+'\n')
                if motion_config.get('output_view','tracks')=='tracks':output=draw_direction(output,direction_config,rows)
            if zebra:
                rows = zebra.step(visible,result.timestamp,result.color_rgb.shape[:2],
                    registration_valid=not engine.stabilizer.failed and engine.registration_observed,
                    scene_changed=distance>=.45 or metrics['state']=='camera/scene change',
                    valid_mask=engine.stabilizer.valid_mask)
                affine = np.vstack([engine.stabilizer.matrix,[0,0,1]])
                binding = {'source_revision':request['info']['source_revision'],'frame':number,
                           'coordinate_space':'registered first-frame analysis image extent',
                           'current_frame_to_source_pixel_centres':
                               (result.transform_to_source @ np.linalg.inv(affine)).tolist()}
                with (run/'motion-zebra.jsonl').open('a',encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(json.dumps({**row,'source_binding':binding},allow_nan=False)+'\n')
                if motion_config.get('output_view','tracks')=='tracks':
                    output=draw_zebra(output,zebra_region,rows)
            assessments = incidents.step(visible,result.timestamp,result.color_rgb.shape[:2],
                registration_valid=not engine.stabilizer.failed,
                scene_changed=distance>=.45 or metrics['state']=='camera/scene change')
            evidence.record(assessments,engine.last_rgb,engine.stabilizer.matrix,
                result.transform_to_source,number,request['info'].get('source_revision'))
            if incidents.enabled and motion_config.get('output_view','tracks')=='tracks':
                draw_geometry(output,incidents.config)
            activity.writerow(metrics); crossings.writerows(events)
            for key, t in visible:
                tracks.writerow([result.timestamp,key,*t['centre'],*t['box'],t.get('speed'),t.get('speed_unit'),t.get('speed_status'),t.get('speed_window_seconds') if t.get('speed') is not None else None,t.get('model_label'),t.get('confidence'),t.get('feature_count'),t.get('tracking_state')]+[t.get(k) for k in display_columns])
            return output
        def finish():
            # Container duration can disagree with decoded PTS (especially VFR
            # with reordered packets). Build the bounded chart after decoding,
            # using actual frame timing, without retaining every row in RAM.
            activity_file.flush()
            last=None
            with (run/'measurements.csv').open(encoding='utf-8') as measurements:
                for last in csv.DictReader(measurements):pass
            if last is None:raise RuntimeError('No decoded timing measurements were saved.')
            duration=max(.001,float(last['timestamp_seconds'])+float(last['duration_seconds']))
            bins=np.zeros((min(600,max(1,int(np.ceil(duration*4-1e-6)))),3),np.float64)
            with (run/'motion-activity.csv').open(encoding='utf-8') as saved_activity:
                for row in csv.DictReader(saved_activity):
                    index=min(len(bins)-1,int(float(row['timestamp_seconds'])/duration*len(bins)))
                    if row['moving_area_percent']:
                        bins[index,0]+=float(row['moving_area_percent']);bins[index,1]+=1
                    bins[index,2]+=1
            summary=engine.summary()
            summary['processing_runtime'] = runtime
            if direction_review:
                summary['direction']={**direction_review.summary(),'synthetic':source_synthetic,
                                      'coordinate_space':'registered first-frame analysis image extent'}
                write_json(run/'motion-direction.json',summary['direction'])
            if zebra:
                summary['zebra']={**zebra.summary(),'source_revision':request['info']['source_revision'],
                    'synthetic':source_synthetic,
                    'coordinate_space':'registered first-frame analysis image extent',
                    'reference_frame':0}
                write_json(run/'motion-zebra.json',summary['zebra'])
            if provenance:
                summary['source_provenance'] = provenance
            if provenance or incident_config:
                summary['synthetic'] = source_synthetic or bool(incident_config and incident_config.get('synthetic') is True)
            if incidents.enabled:
                summary['incidents']=evidence.finish()
            summary['timing']={'source_metadata_duration_seconds':request['info']['duration'],
                               'decoded_duration_seconds':duration,
                               'timeline_aggregation':'Arithmetic mean over observed frames in each time bin; at most 600 bins.'}
            summary['timeline']=[{'start':i*duration/len(bins),'end':(i+1)*duration/len(bins),
                                  'activity_percent':float(v[0]/v[1]) if v[1] else None,
                                  'observed_frames':int(v[2]),
                                  'measured_frame_fraction':float(v[1]/v[2]) if v[2] else 0}
                                 for i,v in enumerate(bins)]
            (run/'motion-heatmap.png').write_bytes(png_bytes(engine.heatmap()))
            np.save(run/'motion-seconds.npy',engine.heat,allow_pickle=False)
            if engine.peak:
                for name in ('mosaic','mask','tracks','original','background','raw-foreground','shadows','flow'):
                    (run/('motion-'+name+'.png')).write_bytes(png_bytes(engine.peak[name]))
            write_json(run/'motion-summary.json',summary)
            activity_file.flush(); crossing_file.flush(); track_file.flush()
            with zipfile.ZipFile(run/'motion-evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for path in run.glob('motion-*'):
                    check_cancel(run)
                    if path.suffix in ('.png','.csv','.json','.jsonl','.npy'):
                        archive.write(path,path.name)
                for path in run.glob('incident-*.png'):
                    check_cancel(run)
                    archive.write(path,path.name)
                check_cancel(run)
                archive.write(run/'coordinates.json','coordinates.json')
                check_cancel(run)
                archive.write(run/'request.json','request.json')
            return summary
        process_video(run,frame,finish,final_key='motion')


def main():
    run=Path(sys.argv[1]).resolve()
    for _ in range(20):
        try:
            with job_lock():
                run_motion(run)
            return
        except RuntimeError as exc:
            if 'Another video job' not in str(exc):
                raise
            time.sleep(.2)
    update(run,'failed',progress=0,message='Another video worker is already active.')


if __name__ == '__main__':
    main()
