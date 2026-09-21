"""Bounded helmet observations on one image/video preview; no scene/LLM models."""
import csv
import json
from pathlib import Path
import sys
import time
from .jobs import job_lock
from .media import load_image, preview_frame, png_bytes, write_json
from .pipeline import Settings, process_frame
from .worker import Cancelled, check_cancel, check_source, update


def process_run(run):
    request=json.loads((run/'request.json').read_text(encoding='utf-8'))
    source,info,config=request['source'],request['info'],request['helmet']
    try:
        check_cancel(run)
        revision=check_source(source,info.get('source_revision'))
        from traffic_ai.local_models import verified_local_models
        verified_local_models()  # Verify BOTH files before any constructor.
        update(run,'running',progress=.05,message='Checking this frame with the installed vehicle/person and helmet models.')
        if info['kind']=='video':
            original,stamp=preview_frame(source,info,config['timestamp'])
        else:
            original,stamp=load_image(source),0.
        if abs(stamp-config['timestamp'])>1e-6:
            raise ValueError('Decoded frame differs from the selected preview. Reopen the frame before retrying.')
        check_cancel(run);check_source(source,revision)
        result=process_frame(original,Settings(**request['settings']),timestamp=stamp,include_features=False,include_stages=False)
        from traffic_ai.perception import Perception
        from traffic_ai.helmet_observations import observe,draw
        detector=Perception(helmet=True)
        detections=detector.detect(result.color_rgb,track=False)
        check_cancel(run);check_source(source,revision)
        summary=observe(detections,result.color_rgb.shape)
        binding={'media_id':config['media_id'],'source_revision':revision,'timestamp_seconds':stamp,
                 'analysis_size':[result.color_rgb.shape[1],result.color_rgb.shape[0]],
                 'coordinate_space':'display-oriented analysis pixel centres',
                 'transform_to_source_pixel_centres':result.transform_to_source.tolist()}
        from .input_preparation import preparation_record
        summary['input_preparation']=preparation_record(original,result,Settings(**request['settings']),binding)
        summary.update(source_binding=binding,detector=detector.manifest,
                       synthetic=info.get('example',{}).get('synthetic') is True)
        (run/'helmet-original.png').write_bytes(png_bytes(result.color_rgb))
        (run/'processed.png').write_bytes(png_bytes(draw(result.color_rgb,summary)))
        write_json(run/'helmet-observations.json',summary)
        with (run/'helmet-observations.csv').open('w',newline='',encoding='utf-8') as stream:
            fields=['state','label','reason','motorcycle_detection_index','person_detection_index',
                    'head_detection_indices','candidate_head_detection_indices','legal_violation','eligible_for_rules']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for row in summary['observations']:
                writer.writerow({**row,'head_detection_indices':json.dumps(row['head_detection_indices']),
                                 'candidate_head_detection_indices':json.dumps(row['candidate_head_detection_indices'])})
        write_json(run/'coordinates.json',binding)
        check_cancel(run);check_source(source,revision)
        write_json(run/'report.json',{'schema_version':1,'kind':'helmet_observation_frame',
                   'source':Path(source).name,'source_info':info,'helmet':summary})
        check_cancel(run);check_source(source,revision)
        update(run,'complete',progress=1,message='Helmet frame observations ready. Unknowns retained; no legal verdict.')
    except Cancelled:
        if (run/'report.json').exists():(run/'report.json').replace(run/'incomplete-report.json')
        update(run,'cancelled',progress=0,message='Helmet check cancelled. Original footage unchanged.')
    except Exception as exc:
        if (run/'report.json').exists():(run/'report.json').replace(run/'incomplete-report.json')
        update(run,'failed',progress=0,message=f'Helmet check unavailable or failed: {exc}')
        raise


def main():
    run=Path(sys.argv[1]).resolve()
    for attempt in range(20):
        try:
            with job_lock():process_run(run)
            return
        except RuntimeError as exc:
            if 'Another video job' not in str(exc):raise
            time.sleep(.2)
    update(run,'failed',progress=0,message='Another processing worker is active.')


if __name__=='__main__':main()
