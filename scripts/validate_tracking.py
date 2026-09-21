"""Run the real automatic-tracking path and check its encoded evidence.

python -m scripts.validate_tracking --source india-delhi-overhead.mp4 --mode vehicles
python -m scripts.validate_tracking --source highway-motion.mp4 --mode ivp
python -m scripts.validate_tracking --run runs/<existing-run>
"""
import argparse
import csv
import json
import time
import zipfile
from itertools import zip_longest
from pathlib import Path
from ivp.jobs import ROOT,job_lock,new_run
from ivp.media import probe_video,video_frames,write_json
from ivp.motion_worker import run_motion
from ivp.pipeline import Settings


def verify(run):
    request=json.loads((run/'request.json').read_text(encoding='utf8'))
    source=Path(request['source']);info=probe_video(source);output=probe_video(run/'processed.mp4')
    max_error=0.;frames=0
    for left,right in zip_longest(video_frames(source,info),video_frames(run/'processed.mp4',output)):
        assert left is not None and right is not None,'Exported frame count changed'
        max_error=max(max_error,abs(left[1]-right[1]));frames+=1
    assert max_error<.002 and output['has_audio']==info['has_audio'] and output['codec']=='h264'
    with zipfile.ZipFile(run/'motion-evidence.zip') as archive:
        assert archive.testzip() is None
        files=len(archive.namelist())
    identities=set();sample_count=speeds=0
    with (run/'motion-tracks.csv').open(encoding='utf8') as f:
        for row in csv.DictReader(f):
            assert row['track_id']!='SELECTED','Automatic test unexpectedly used a manual target'
            identities.add(row['track_id']);sample_count+=1;speeds+=bool(row['speed'])
    summary=json.loads((run/'motion-summary.json').read_text(encoding='utf8'))
    result={'run':run.name,'source':source.name,'mode':request['motion'].get('tracking_mode','ivp'),
        'decoded_frames':frames,'max_timestamp_error_seconds':max_error,'evidence_files':files,
        'track_ids':len(identities),'observed_track_samples':sample_count,'accepted_speed_samples':speeds,
        'speed_unit':summary['speed']['unit'],'camera_registration_rejections':summary['stabilization']['rejected_frames'],
        'note':'Export and diagnostic verification, not exhaustive detection/tracking accuracy or real-world speed validation.'}
    if source.name in ('india-city.mp4','india-intersection.mp4'):
        from scripts.check_city_identities import check
        spot=check(run,'city-tracking-check.json' if source.name=='india-city.mp4' else 'intersection-tracking-check.json')
        result['identity_spot_check']={k:spot[k] for k in ['matched_observations','total_observations','objects_with_identity_changes']}
    elif source.name=='india-delhi-overhead.mp4':
        from scripts.check_delhi_tracking import check
        spot=check(run)
        result['identity_spot_check']={k:spot[k] for k in ['matched_observations','total_observations','objects_with_identity_changes']}
        result['landmark_registration_check']=spot['registration']
    write_json(run/'automatic-validation.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    parser.add_argument('--source',default='india-city.mp4',choices=['india-city.mp4','india-delhi-overhead.mp4','india-intersection.mp4','highway-motion.mp4'])
    parser.add_argument('--mode',default='vehicles',choices=['vehicles','ivp']);args=parser.parse_args()
    run=args.run
    if run is None:
        source=ROOT/'demo'/args.source
        config={'axis':'horizontal','position':.7 if args.source=='india-city.mp4' else .65,
                'rectangle':[.04,.28,.94,.98] if args.source=='india-city.mp4' else [0,.4,1,.98],
                'tracking_mode':args.mode,'output_view':'tracks'}
        start=time.monotonic()
        with job_lock():
            run=new_run();print('Run:',run.name,flush=True)
            write_json(run/'request.json',{'source':str(source),'info':probe_video(source),
                'settings':Settings(max_width=960).to_dict(),'selected_masks':[],'motion':config})
            run_motion(run)
        print('Processing seconds:',round(time.monotonic()-start,2),flush=True)
    print(json.dumps(verify(run),indent=2),flush=True)


if __name__=='__main__':main()
