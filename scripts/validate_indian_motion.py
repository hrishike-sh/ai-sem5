"""Reproduce the three sourced Indian-video runs without loading any AI model.

Run from the project root: python -m scripts.validate_indian_motion
See demo/motion-sources.json for source URLs; videos are local-only.
"""
import json
import time
from pathlib import Path

from ivp.jobs import ROOT, job_lock, new_run
from ivp.media import probe_video, write_json, video_frames
from ivp.pipeline import Settings
from ivp.motion_worker import run_motion


def main():
    rows=[]
    for name in ['india-delhi-overhead','india-intersection','india-rickshaws']:
        source=ROOT/'demo'/(name+'.mp4')
        if not source.is_file():
            raise SystemExit(f'Missing {source}. Download the credited source in demo/motion-sources.json first.')
        info=probe_video(source)
        with job_lock():
            run=new_run(); start=time.monotonic()
            config={'axis':'horizontal' if 'overhead' in name else 'vertical',
                    'position':.65 if 'overhead' in name else .5,'rectangle':[0,.4,1,.98]}
            write_json(run/'request.json',{'source':str(source),'info':info,
                'settings':Settings(max_width=960).to_dict(),'selected_masks':[],'motion':config})
            run_motion(run)
        report=json.loads((run/'report.json').read_text(encoding='utf8'))
        output=probe_video(run/'processed.mp4')
        source_stamps=[t for _,t,_,_ in video_frames(source,info)]
        output_stamps=[t for _,t,_,_ in video_frames(run/'processed.mp4',output)]
        assert len(source_stamps)==len(output_stamps)
        max_error=max(abs(a-b) for a,b in zip(source_stamps,output_stamps))
        assert max_error<.002
        row={'source':name,'run':run.name,'elapsed_seconds':round(time.monotonic()-start,2),
             'frames':len(output_stamps),'duration':output['duration'],'max_timestamp_error_seconds':max_error,
             'motion':report['motion']}
        rows.append(row)
        write_json(ROOT/'runs'/'indian-validation.json',rows)
        print(json.dumps({k:v for k,v in row.items() if k!='motion'}),flush=True)


if __name__=='__main__':
    main()
