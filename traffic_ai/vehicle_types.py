"""Optional pinned UVH-26 vehicle subtype proposals, never a vehicle census."""
import hashlib
import math
import threading
import numpy as np
from .runtime import configure_cache, model_path, runtime_root

REPOSITORY = 'iisc-aim/UVH-26'
REVISION = '4a22412775adb6f97f22735647afee976b4638a0'
SHA256 = '12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150'
BYTES = 57142241
CLASSES = {str(i): label for i, label in enumerate(('Hatchback','Sedan','SUV','MUV','Bus','Truck',
           'Three-wheeler','Two-wheeler','LCV','Mini-bus','tempo-traveller','bicycle','Van','Others'))}
THRESHOLDS = {'inference_floor': .25, 'candidate_score': .5, 'match_iou': .3}
SCORE_NOTICE = 'Raw detector scores are not calibrated probabilities or measured accuracy. Thresholds are heuristic.'
LIMITATIONS = ['Current-image detector proposals only; no unique vehicle or video traffic counts.',
               'No detection does not mean absence. Small, occluded and distant vehicles can be missed.',
               'Subtype labels and geometric box matches are candidates requiring visual review.',
               'Two-wheeler training boxes may include the rider. Three-wheeler and other subtypes can be confused.',
               'Candidate counts include detector proposals, not manually verified objects.']
COMPLETE = 'Pinned local vehicle subtype inference completed.'
UNAVAILABLE = 'Pinned local vehicle subtype inference is unavailable; no download was attempted.'
_LOCK = threading.RLock()
_CACHE = {}


def _metadata():
    return {'name': REPOSITORY, 'revision': REVISION, 'sha256': SHA256, 'classes': dict(CLASSES),
            'preprocessing': 'Bound original RGB converted to contiguous BGR; CPU, imgsz=640, max_det=100; boxes in original-source pixels.'}


def predict_types(rgb):
    with _LOCK:
        configure_cache()
        path = model_path('vehicle_types', runtime_root()/'models'/'uvh26-yolo11s-4a224127.pt')
        if not path.is_file() or path.stat().st_size != BYTES:
            raise RuntimeError('Pinned vehicle subtype weights are not available locally.')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != SHA256:
            raise RuntimeError('Vehicle subtype weights do not match their pin.')
        key = (str(path), digest)
        model = _CACHE.get(key)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(path))
            if {str(k):str(v) for k,v in model.names.items()} != CLASSES:
                raise RuntimeError('Vehicle subtype model has unsupported classes.')
            _CACHE.clear()
            _CACHE[key] = model
        result = model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]), device='cpu', imgsz=640,
                               conf=.25, max_det=100, verbose=False)[0]
        if result.boxes is None:
            return []
        return [{'class_id':int(cls), 'label':CLASSES[str(int(cls))], 'model_score':float(score),
                 'box':list(map(float,box))} for cls,score,box in zip(result.boxes.cls.cpu().tolist(),
                  result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]


def _iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0


def _status(candidates):
    if not candidates:
        return 'not_observed'
    if len({d['class_id'] for d in candidates}) > 1:
        return 'unclear'
    if any(_iou(a['box'],b['box']) < .3 for i,a in enumerate(candidates) for b in candidates[i+1:]):
        return 'unclear'
    return 'candidate' if max(d['model_score'] for d in candidates)>=.5 else 'low_confidence'


def _counts(detections):
    counts = {label:0 for label in CLASSES.values()}
    for detection in detections:
        if detection['model_score'] >= .5:
            counts[detection['label']] += 1
    return {'scope':'current_image_detector_proposals', 'by_type':counts, 'total':sum(counts.values()),
            'score_floor':.5, 'maximum_detections':100}


def _valid_detection(d):
    if (not isinstance(d,dict) or type(d.get('class_id')) is not int or str(d['class_id']) not in CLASSES
            or d.get('label') != CLASSES[str(d['class_id'])]
            or type(d.get('model_score')) not in (int,float) or not math.isfinite(d['model_score'])
            or not .25 <= d['model_score'] <= 1):
        return False
    box = d.get('box')
    return (isinstance(box,list) and len(box)==4
            and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100000 for v in box)
            and box[0]<box[2] and box[1]<box[3])


def build_evidence(rgb, subjects, image_width, image_height, *, truncated=False, detections=None):
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or type(image_width) is not int or type(image_height) is not int or min(image_width,image_height)<=0):
        raise ValueError('Vehicle subtype evidence requires bound original RGB and positive preview dimensions.')
    result = {'version':1,'state':'complete','reason':COMPLETE,'model':_metadata(),'score_notice':SCORE_NOTICE,
              'thresholds':dict(THRESHOLDS),'subjects':[{'id':s['id'], 'status':'not_applicable' if s['kind']=='person' else 'not_observed',
                                                       'candidates':[]} for s in subjects],
              'unassigned_detections':[], 'candidate_counts':_counts([]),'limitations':list(LIMITATIONS)}
    try:
        raw_detections = predict_types(rgb) if detections is None else detections
    except (RuntimeError,OSError,ImportError,ValueError):
        result.update(state='unavailable',reason=UNAVAILABLE)
        for row,s in zip(result['subjects'],subjects):
            if s['kind']!='person': row['status']='unavailable'
        return result
    sx,sy=rgb.shape[1]/image_width,rgb.shape[0]/image_height
    vehicles=[(i,[s['bbox'][0]*sx,s['bbox'][1]*sy,s['bbox'][2]*sx,s['bbox'][3]*sy])
              for i,s in enumerate(subjects) if s['kind']!='person']
    all_detections=[]
    for raw in raw_detections[:100]:
        if not _valid_detection(raw) or raw['box'][2]>rgb.shape[1] or raw['box'][3]>rgb.shape[0]:
            continue
        d={key:raw[key] for key in ('class_id','label','model_score','box')}
        d['box']=list(map(float,d['box']))
        d['model_score']=float(d['model_score'])
        all_detections.append(d)
        matches=[i for i,box in vehicles if _iou(d['box'],box)>=.3]
        if len(matches)==1:
            i=matches[0]
            result['subjects'][i]['candidates'].append({**d,'evidence_id':subjects[i]['evidence_ids'][-1]})
        else:
            result['unassigned_detections'].append({**d,'reason':'ambiguous_vehicle_overlap' if matches else 'no_numbered_vehicle_match'})
    for row,s in zip(result['subjects'],subjects):
        if s['kind']!='person': row['status']=_status(row['candidates'])
    result['candidate_counts']=_counts(all_detections)
    return result


def validate_vehicle_evidence(evidence, subjects):
    def require(condition):
        if not condition: raise ValueError('Saved vehicle subtype evidence is invalid or mismatched.')
    require(isinstance(evidence,dict) and set(evidence)=={'version','state','reason','model','score_notice','thresholds','subjects','unassigned_detections','candidate_counts','limitations'})
    require(type(evidence['version']) is int and evidence['version']==1 and evidence['state'] in ('complete','unavailable'))
    require(evidence['reason']==(COMPLETE if evidence['state']=='complete' else UNAVAILABLE))
    require(evidence['model']==_metadata() and evidence['score_notice']==SCORE_NOTICE and evidence['thresholds']==THRESHOLDS and evidence['limitations']==LIMITATIONS)
    require(isinstance(evidence['subjects'],list) and len(evidence['subjects'])==len(subjects))
    detections=[]
    for row,s in zip(evidence['subjects'],subjects):
        require(isinstance(row,dict) and set(row)=={'id','status','candidates'} and row['id']==s['id'] and isinstance(row['candidates'],list))
        for d in row['candidates']:
            require(_valid_detection(d) and set(d)=={'class_id','label','model_score','box','evidence_id'} and d['evidence_id']==s['evidence_ids'][-1])
        detections.extend(row['candidates'])
        if s['kind']=='person':
            require(row['status']=='not_applicable' and not row['candidates'])
        elif evidence['state']=='unavailable':
            require(row['status']=='unavailable' and not row['candidates'])
        else:
            require(row['status']==_status(row['candidates']))
    require(isinstance(evidence['unassigned_detections'],list))
    for d in evidence['unassigned_detections']:
        require(_valid_detection(d) and set(d)=={'class_id','label','model_score','box','reason'} and d['reason'] in ('ambiguous_vehicle_overlap','no_numbered_vehicle_match'))
    detections.extend(evidence['unassigned_detections'])
    require(len(detections)<=100 and evidence['candidate_counts']==_counts(detections))
    require(evidence['state']=='complete' or not detections)
    return evidence
