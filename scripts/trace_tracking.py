"""Record detector/association evidence without changing app tracking behavior.

Outputs stay under the existing H-drive experiment directory. No downloads.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ivp.jobs import ROOT, job_lock
from ivp.media import probe_video, write_json
from ivp.motion_worker import run_motion
from ivp.pipeline import Settings
from traffic_ai.runtime import configure_cache
from traffic_ai.vehicle_motion import VehicleTracker
from scripts.validate_tracking import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='india-delhi-overhead.mp4')
    parser.add_argument('--anchor', action='store_true', help='Use the current ground-anchor association; default traces legacy ByteTrack for comparison')
    parser.add_argument('--weak-recovery', action='store_true', help='Offline trial: repeated observed weak detections may recover recent IDs')
    parser.add_argument('--robust-recovery', action='store_true', help='Offline trial: robust observed geometry for sustained prediction drift')
    args = parser.parse_args()
    source = ROOT/'demo'/args.source
    if source.name != args.source or not source.is_file():
        raise ValueError('Choose an existing local demo filename.')
    folder = configure_cache()/'experiments'/('association-trace-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True)
    info = probe_video(source)
    original_load, original_update = VehicleTracker._load, VehicleTracker.update
    trace = (folder/'association.jsonl').open('w', encoding='utf-8')
    context = {}

    def update(self, boxes, stamp, *args):
        context['timestamp'] = stamp
        return original_update(self, boxes, stamp, *args)

    def load(self):
        original_load(self)
        if args.robust_recovery:
            from traffic_ai.robust_recovery import RobustObservedRecovery
            self.association = RobustObservedRecovery(self.association.args)
            self.manifest['experimental_association'] = 'ground anchor + repeated weak boxes + robust observed geometry'
        elif args.weak_recovery:
            from traffic_ai.recovery import RepeatedWeakRecovery
            self.association = RepeatedWeakRecovery(self.association.args)
            self.manifest['experimental_association'] = 'ground anchor + repeated observed weak-box recovery'
        elif args.anchor:
            from traffic_ai.association import GroundAnchorTracker
            self.association = GroundAnchorTracker(self.association.args)
            self.manifest['experimental_association'] = 'overlap + bottom-centre continuity'
        else:
            from ultralytics.trackers.byte_tracker import BYTETracker
            self.association = BYTETracker(self.association.args)
            self.manifest['experimental_association'] = 'legacy ByteTrack overlap-only comparison'
        original_associate = self.association.update
        def snapshot(tracks):
            return [{'id':int(t.track_id), 'box':t.xyxy.tolist(), 'state':int(t.state),
                     'active':bool(t.is_activated), 'score':float(t.score)} for t in tracks]
        def associate(results, image, *args, **kwargs):
            before = snapshot(self.association.tracked_stracks+self.association.lost_stracks)
            rows = original_associate(results, image, *args, **kwargs)
            trace.write(json.dumps({'timestamp':context['timestamp'], 'tracker_frame_id':self.association.frame_id,
                                    'detections':results.data.tolist(),
                                    'before':before, 'rows':rows.tolist(),
                                    'after':snapshot(self.association.tracked_stracks+self.association.lost_stracks)})+'\n')
            t=context['timestamp']
            if abs(t*2-round(t*2)) < .001:
                frame=image.copy()
                for row in rows:
                    x,y,x2,y2,key,score=row[:6]
                    colour=tuple(int(c) for c in np.random.default_rng(int(key)).integers(80,255,3))
                    cv2.rectangle(frame,(int(x),int(y)),(int(x2),int(y2)),colour,2)
                    cv2.putText(frame,f'{int(key)} {score:.2f}',(int(x),max(15,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1)
                cv2.imwrite(str(folder/f'tracks-{t:04.1f}.jpg'),frame[:,:,::-1])
            return rows
        self.association.update = associate
    VehicleTracker._load, VehicleTracker.update = load, update
    config = {'axis':'horizontal','position':.7 if source.name=='india-city.mp4' else .65,
              'rectangle':[.04,.28,.94,.98] if source.name=='india-city.mp4' else [0,.4,1,.98],
              'tracking_mode':'vehicles','output_view':'tracks'}
    write_json(folder/'request.json',{'source':str(source),'info':info,'settings':Settings(max_width=960).to_dict(),
                                      'selected_masks':[],'motion':config})
    write_json(folder/'experiment.json', {'variant':'robust_observed_recovery' if args.robust_recovery else 'repeated_weak_recovery' if args.weak_recovery else 'ground_anchor' if args.anchor else 'legacy_bytetrack',
                                        'production_default_changed':False})
    print(folder, flush=True)
    try:
        with job_lock():
            run_motion(folder)
    finally:
        trace.close()
        VehicleTracker._load, VehicleTracker.update = original_load, original_update
    print(json.dumps(verify(folder),indent=2),flush=True)


if __name__ == '__main__':
    main()
