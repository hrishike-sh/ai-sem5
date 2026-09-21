"""Run fixture videos through the real localhost upload / detector / export pipeline.

The fixture's evaluation_only directory is deliberately never opened by this runner.
It writes new app runs and a fresh acceptance record; fixture bytes are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ivp.media import probe_video, video_frames, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('fixtures',type=Path)
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--url',default='http://127.0.0.1:8511')
    args=parser.parse_args()
    html=urllib.request.urlopen(args.url,timeout=15).read().decode()
    token=re.search(r'name="traffic-token" content="([^"]+)"',html)[1]
    def api(path,data=None,raw=None):
        body=raw if raw is not None else json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(args.url+path,data=body,headers={'X-Traffic-Token':token})
        with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
    checkpoint=ROOT/'runs'/('incident-fixtures-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'.json')
    record={'synthetic':True,'fixture_root':str(args.fixtures.resolve()),'ground_truth_used_for_tracking':False,
        'mapping':'Fixture incidents unchanged; zone_role defaults to observe. Runtime upload identity added to calibration; tracking_mode=vehicles, rectangle=[.1,.1,.9,.98], output_view=tracks.',
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'ivp/incidents.py',ROOT/'ivp/stop_evidence.py',ROOT/'ivp/incident_export.py',ROOT/'ivp/motion_worker.py',ROOT/'web_server.py',ROOT/'web/app.js']},'cases':[]}
    print('ACCEPTANCE',checkpoint,flush=True)
    for config_path in sorted((args.fixtures/'configs').glob('*.json')):
        name=config_path.stem
        if args.only and name not in args.only:continue
        cfg=json.loads(config_path.read_text())
        source=args.fixtures/cfg['media']
        entry={'scenario':name,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        record['cases'].append(entry)
        media=api('/api/upload?name='+urllib.parse.quote(source.name),raw=source.read_bytes())
        calibration=cfg['calibration']
        if calibration is not None:
            calibration={**calibration,'media_id':media['id'],'source_revision':media['source_revision'],'reference_frame':0}
        payload={'media':media['id'],'tracking_mode':'vehicles','output_view':'tracks',
            'rectangle':[.1,.1,.9,.98],'axis':'horizontal','position':.65,
            'calibration':calibration,'incidents':cfg['incidents']}
        try:
            run=api('/api/motion',payload)
        except urllib.error.HTTPError as exc:
            entry.update(http_status=exc.code,rejected=json.loads(exc.read()).get('error'))
            print(name,'REJECTED',entry['rejected'],flush=True);write_json(checkpoint,record);continue
        entry['run_id']=run['id'];write_json(checkpoint,record)
        deadline=time.monotonic()+300
        while run['status']['state'] in ('queued','running','encoding','unknown') and time.monotonic()<deadline:
            time.sleep(2);run=api('/api/run?id='+run['id'])
        entry['status']=run['status']
        if run['status']['state']=='complete':
            report=run['report']['motion'];entry['tracking']=report['tracking'];entry['incidents']=report['incidents']
            path=ROOT/'runs'/run['id']/'processed.mp4'
            entry['output']=probe_video(path)
            original_times=[round(t,6) for _,t,_,_ in video_frames(source,probe_video(source))]
            exported_times=[round(t,6) for _,t,_,_ in video_frames(path,entry['output'])]
            entry['timing_matches']=original_times==exported_times
            import zipfile
            with zipfile.ZipFile(path.parent/'motion-evidence.zip') as archive:
                entry['zip_crc_ok']=archive.testzip() is None
                entry['zip_incident_files']=[n for n in archive.namelist() if 'incident' in n]
            print(name,run['id'],report['incidents']['counts'],flush=True)
        else:print(name,run['status'],flush=True)
        write_json(checkpoint,record)
    print('DONE',checkpoint,flush=True)


if __name__=='__main__':main()
