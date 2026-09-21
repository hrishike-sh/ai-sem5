"""Offline speed-observation experiment over exported, observed trajectories."""
import argparse
import csv
import json
from pathlib import Path

from ivp.speed import SpeedEstimator
from ivp.media import write_json


class ReferenceSpeedEstimator(SpeedEstimator):
    """Ablation: retain current cadence rules, disable only the position gate."""
    def _position_jump(self, history):
        return False


def replay(run, estimator_class):
    request=json.loads((run/'request.json').read_text())
    interval=1/(request['info'].get('fps') or 30)
    estimator=estimator_class(request['motion'].get('calibration'),frame_interval=interval)
    size=json.loads((run/'coordinates.json').read_text())['processed_size']
    tracks={};changes=[];accepted=0
    with (run/'motion-tracks.csv').open() as f:
        for row in csv.DictReader(f):
            key=row['track_id'];stamp=float(row['timestamp_seconds'])
            track=tracks.setdefault(key,{})
            if (stamp-track.get('last_observed',stamp)>estimator.timing.recovery
                    or row.get('tracking_state')=='recovered from weak observations'):
                track.pop('speed_history',None)
            track['last_observed']=stamp
            track['box']=tuple(float(row[k]) for k in ('bbox_x','bbox_y','bbox_width','bbox_height'))
            status=row['speed_status']
            track['speed_blocked']=status if status in ('vehicle clipped by frame','weak detection','vehicle touches a camera-warp border','overlapping vehicle hypotheses') else None
            estimator.update(track,stamp,tuple(reversed(size)))
            accepted+=track['speed'] is not None
            old=None if not row['speed'] else float(row['speed'])
            if old!=track['speed']:
                changes.append({'timestamp':stamp,'track_id':key,'before':old,'after':track['speed'],'status':track['speed_status']})
    return {'accepted_samples':accepted,'changed_samples':len(changes),'changes':changes}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    results={'reference_without_position_gate':replay(args.run,ReferenceSpeedEstimator),
             'with_position_gate':replay(args.run,SpeedEstimator)}
    write_json(args.run/'speed-replay-local-consistency.json',results)
    print(json.dumps({k:{n:v for n,v in r.items() if n!='changes'} for k,r in results.items()},indent=2))


if __name__=='__main__':main()
