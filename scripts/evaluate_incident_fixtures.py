"""Post-run acceptance only: compare inference exports to independent fixture truth."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ivp.media import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('record',type=Path);args=parser.parse_args()
    record=json.loads(args.record.read_text())
    fixture=Path(record['fixture_root'])
    manifest=json.loads((fixture/'manifest.json').read_text())
    scenarios={s['id']:s for s in manifest['scenarios']}
    results=[]
    for case in record['cases']:
        name=case['scenario'];spec=scenarios[name];checks={}
        checks['input_hash']=case['input_sha256']==spec['video_sha256']
        if name=='speed_36_invalid_calibration':
            checks['invalid_calibration_rejected']=case.get('http_status')==400
            results.append({'scenario':name,'checks':checks,'passed':all(checks.values())});continue
        checks['complete']=case.get('status',{}).get('state')=='complete'
        checks['timing']=case.get('timing_matches',False)
        checks['zip']=case.get('zip_crc_ok',False)
        if not checks['complete']:
            results.append({'scenario':name,'checks':checks,'passed':False});continue
        run=ROOT/'runs'/case['run_id']
        events=[json.loads(line) for line in (run/'motion-incidents.jsonl').read_text().splitlines()]
        line=[e for e in events if e['type']=='line_crossing' and e['status']=='candidate']
        line_assessments=[e for e in events if e['type']=='line_crossing']
        speeding=[e for e in events if e['type']=='speeding' and e['status']=='candidate']
        checks['actual_track_observed']=case['tracking']['confirmed_tracks']>0
        crossing=spec['expected']['line_outcome']=='candidate'
        checks['line_candidate_expected']=bool(line)==crossing
        if crossing and line:
            window=spec['expected']['crossing_event_window_seconds']
            checks['line_timing']=len(line)==1 and window[0]<=line[0]['timestamp_seconds']<=window[1]
        if spec['expected'].get('crossing_time_seconds') is not None and not spec['occlusion_half_open_seconds']:
            window=spec['expected']['crossing_event_window_seconds']
            checks['crossing_assessed_within_window']=len(line_assessments)==1 and window[0]<=line_assessments[0]['timestamp_seconds']<=window[1]
        if name=='unknown_cross_18':
            checks['unknown_explicitly_withheld']=len(line_assessments)==1 and line_assessments[0]['status']=='withheld'
        if name=='stop_before_line_18':checks['no_crossing']=not line_assessments
        if name=='stop_required_wait_18':
            checks['completed_stop_observed']=len(line_assessments)==1 and line_assessments[0]['status']=='observation'
            proof=line_assessments[0]['measurements'].get('stop_evidence') if line_assessments else None
            checks['observed_stop_window_exported']=bool(proof and proof['duration_seconds']>=1-1e-6 and
                proof['observed_samples']>=8 and proof['position_diameter_bound_m']<=proof['maximum_position_diameter_m'] and
                proof['fitted_speed_kmh']<=proof['maximum_fitted_speed_kmh'])
        checks['speed_candidate_expected']=bool(speeding) if name=='speed_above_36' else not speeding if name!='speed_36_occluded' else True
        checks['synthetic_labels']=all(e['synthetic'] is True for e in events)
        checks['neutral_zebra_zone']=all(e['status']!='candidate' for e in events if e['type']=='zone_entry')
        rows=list(csv.DictReader((run/'motion-tracks.csv').open()))
        truth=list(csv.DictReader((fixture/spec['ground_truth']).open()))
        by_frame={int(r['frame']):r for r in truth}
        errors=[]
        for row in rows:
            gt=by_frame.get(round(float(row['timestamp_seconds'])*manifest['media']['fps']))
            if (gt and gt['eligible_observation']=='True' and gt['within_calibration']=='True' and row['speed'] and
                    row['speed_unit']=='km/h' and row['speed_status']=='estimated'):
                errors.append(abs(float(row['speed'])-float(gt['declared_simulated_kmh'])))
        if spec['occlusion_half_open_seconds']:
            start,end=spec['occlusion_half_open_seconds']
            checks['no_speed_through_gap_or_initial_recovery']=not any(
                r['speed'] and start<=float(r['timestamp_seconds'])<end+.5 for r in rows)
            checks['no_speed_candidate_during_gap']=not any(start<=e['timestamp_seconds']<end+.5 for e in speeding)
        results.append({'scenario':name,'run_id':case['run_id'],'checks':checks,'passed':all(checks.values()),
            'counts':case['incidents']['counts'],'candidate_times':[{'type':e['type'],'time':e['timestamp_seconds']} for e in events if e['status']=='candidate'],
            'accepted_speed_error_samples':len(errors),'median_abs_speed_error_kmh':statistics.median(errors) if errors else None,
            'max_abs_speed_error_kmh':max(errors) if errors else None,
            'note':'Errors include speed-window lag during stopping/starting. Single synthetic vehicle, not a real-traffic accuracy benchmark.'})
    output=args.record.with_name(args.record.stem+'-evaluation.json')
    result={'synthetic':True,'input_record':str(args.record),'record_sha256':hashlib.sha256(args.record.read_bytes()).hexdigest(),
        'evaluation_only':True,'ground_truth_used_after_inference_only':True,'cases':results,
        'passed':sum(r['passed'] for r in results),'total':len(results)}
    write_json(output,result)
    print(json.dumps(result,indent=2))
    print('EVALUATION',output)


if __name__=='__main__':main()
