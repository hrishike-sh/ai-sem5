"""Bounded local plate candidates. Text never establishes an identity or violation."""
import copy
import hashlib
import math
import re
import threading

import numpy as np

from .runtime import configure_cache, runtime_root

DETECTOR_REVISION = '251a30d7daedca065f56e04b0af04052c907c68f'
DETECTOR_SHA256 = '0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2'
OCR_SHA256 = 'e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8'
THRESHOLDS = {'detector_floor': .25, 'candidate_detector': .5, 'candidate_ocr': .8,
              'plate_containment': .9, 'max_detections': 20, 'minimum_crop_width': 32,
              'minimum_crop_height': 12}
SCORE_NOTICE = 'Raw detector and OCR scores are not calibrated probabilities or measured accuracy. Thresholds are heuristic.'
LIMITATIONS = [
    'Plate text is a visual candidate, never proof of ownership, registration validity or a violation.',
    'No owner lookup, identity lookup or traffic offence is inferred from a plate number.',
    'Small, blurred, angled, occluded and multi-line plates may be missed or misread; no detection does not mean absence.',
    'English OCR only; original OCR is retained separately from a manual transcription.',
    'The detector publisher reports training/test contamination; published accuracy is not a reliable field estimate.',
    'At most 20 current-image plate proposals are processed; this is not a plate or vehicle census.'
]
COMPLETE = 'Local plate candidate analysis completed; readings require visual review.'
UNAVAILABLE = 'Pinned local plate detection or OCR is unavailable; no download was attempted.'
_LOCK = threading.RLock()
_CACHE = {}


def _metadata():
    return {'detector': {'name': 'morsetechlab/yolov11-license-plate-detection',
                         'revision': DETECTOR_REVISION, 'sha256': DETECTOR_SHA256, 'license': 'AGPL-3.0'},
            'ocr': {'name': 'JaidedAI/EasyOCR english_g2', 'version': '1.7.2',
                    'revision': 'c4f3cd7225efd4f85451bd8b4a7646ae9a092420',
                    'sha256': OCR_SHA256, 'license': 'Apache-2.0'},
            'preprocessing': 'RGB to BGR detector; CPU imgsz=960. Source-pixel boxes. Greyscale crop OCR, max 640x160; English alphanumeric allowlist.'}


def _check_file(path, expected):
    if not path.is_file():
        raise RuntimeError('Pinned local plate weights missing.')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise RuntimeError('Local plate weights differ from their pin.')


def predict_plates(rgb):
    """No downloader, network call or remote-code model loader is used here."""
    with _LOCK:
        configure_cache()
        folder = runtime_root() / 'models' / 'plates'
        detector_path, ocr_path = folder / 'detector.pt', folder / 'english_g2.pth'
        _check_file(detector_path, DETECTOR_SHA256)
        _check_file(ocr_path, OCR_SHA256)
        key = str(folder)
        if key not in _CACHE:
            from ultralytics import YOLO
            import easyocr
            if easyocr.__version__ != '1.7.2':
                raise RuntimeError('Unsupported OCR package version.')
            detector = YOLO(str(detector_path))
            if len(detector.names) != 1 or 'plate' not in str(detector.names[0]).lower():
                raise RuntimeError('Unsupported plate model classes.')
            reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(folder),
                                    user_network_directory=str(folder / 'user_network'),
                                    download_enabled=False, detector=False, verbose=False)
            _CACHE.clear()
            _CACHE[key] = detector, reader
        detector, reader = _CACHE[key]
        import cv2
        result = detector.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]), device='cpu',
                                  imgsz=960, conf=.25, max_det=20, verbose=False)[0]
        if result.boxes is None:
            return []
        records = []
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            x1,y1,x2,y2 = box
            record = {'box': list(map(float, box)), 'detector_score': float(confidence), 'ocr_score': None, 'text': ''}
            if x2-x1 >= 32 and y2-y1 >= 12:
                crop = rgb[max(0,int(y1)):min(rgb.shape[0],math.ceil(y2)),max(0,int(x1)):min(rgb.shape[1],math.ceil(x2))]
                grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                scale=min(1,640/grey.shape[1],160/grey.shape[0])
                if scale < 1:
                    grey = cv2.resize(grey,(max(1,round(grey.shape[1]*scale)),max(1,round(grey.shape[0]*scale))))
                output = reader.recognize(grey, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ', detail=1,
                                          paragraph=False, batch_size=1, workers=0)
                if (isinstance(output,list) and len(output)==1 and isinstance(output[0],(list,tuple))
                        and len(output[0])==3 and type(output[0][2]) in (int,float,np.float32,np.float64)):
                    raw_text, score = output[0][1], float(output[0][2])
                    # Reject unsupported/overlong output instead of silently fabricating a shortened plate.
                    if _text_valid(raw_text) and math.isfinite(score) and 0 <= score <= 1:
                        record.update(text=raw_text, ocr_score=score)
            records.append(record)
        return records


def _text_valid(text):
    return isinstance(text,str) and len(text) <= 16 and re.fullmatch(r'[A-Z0-9 ]*',text) is not None


def _score(value):
    return type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1


def _valid_detection(d, width, height):
    if (not isinstance(d,dict) or not {'box','text','ocr_score','detector_score'} <= set(d)
            or not _score(d.get('detector_score')) or d['detector_score'] < .25):
        return False
    if not _text_valid(d.get('text')) or (d.get('ocr_score') is not None and not _score(d['ocr_score'])):
        return False
    if d.get('ocr_score') is None and d['text']:
        return False
    box=d.get('box')
    return (isinstance(box,list) and len(box)==4 and all(type(v) in (int,float) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def _matches(d, subjects, source):
    plate=d['box']; area=(plate[2]-plate[0])*(plate[3]-plate[1])
    sx,sy=source['width']/source['preview_width'],source['height']/source['preview_height']
    matches=[]
    for i,s in enumerate(subjects):
        if s['kind']=='person': continue
        b=s['bbox']; box=[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy]
        intersection=max(0,min(plate[2],box[2])-max(plate[0],box[0]))*max(0,min(plate[3],box[3])-max(plate[1],box[1]))
        if intersection/area >= .9: matches.append(i)
    return matches


def _status(candidates):
    if not candidates: return 'not_observed'
    if len(candidates)>1: return 'unclear'
    d=candidates[0]
    if not d['text'].strip() or d['ocr_score'] is None: return 'unreadable'
    if len(d['text'].replace(' ','')) < 4 or d['detector_score'] < .5 or d['ocr_score'] < .8: return 'unclear'
    return 'candidate'


def build_evidence(rgb, subjects, image_width, image_height, *, truncated=False, detections=None):
    if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or min(rgb.shape[:2])<=0 or type(image_width) is not int or type(image_height) is not int
            or min(image_width,image_height)<=0):
        raise ValueError('Plate evidence requires original RGB and positive preview dimensions.')
    result={'version':1,'state':'complete','reason':COMPLETE,'model':_metadata(),
            'source':{'width':rgb.shape[1],'height':rgb.shape[0],'preview_width':image_width,'preview_height':image_height},
            'score_notice':SCORE_NOTICE,'thresholds':dict(THRESHOLDS),'limitations':list(LIMITATIONS),
            'subjects':[{'id':s['id'],'status':'not_applicable' if s['kind']=='person' else 'not_observed',
                         'candidates':[],'manual_correction':None} for s in subjects], 'unassigned_detections':[]}
    try:
        raw = predict_plates(rgb) if detections is None else detections
        if not isinstance(raw,list):
            raise RuntimeError('Unsupported plate detector output.')
    except (RuntimeError,OSError,ImportError,ValueError):
        result.update(state='unavailable',reason=UNAVAILABLE)
        for row,s in zip(result['subjects'],subjects):
            if s['kind']!='person': row['status']='unavailable'
        return result
    for item in raw[:20]:
        if not _valid_detection(item,rgb.shape[1],rgb.shape[0]): continue
        d={key:copy.deepcopy(item[key]) for key in ('box','detector_score','ocr_score','text')}
        d['plate_id']=f'plate-{1+sum(len(r["candidates"]) for r in result["subjects"])+len(result["unassigned_detections"])}'
        matches=_matches(d,subjects,result['source'])
        if len(matches)==1:
            i=matches[0]
            result['subjects'][i]['candidates'].append({**d,'evidence_id':subjects[i]['evidence_ids'][-1]})
        else:
            result['unassigned_detections'].append({**d,'reason':'ambiguous_vehicle_overlap' if matches else 'no_numbered_vehicle_match'})
    for row,s in zip(result['subjects'],subjects):
        if s['kind']!='person': row['status']=_status(row['candidates'])
    return result


def validate_plate_evidence(evidence, subjects):
    def require(condition):
        if not condition: raise ValueError('Saved plate evidence is invalid or mismatched.')
    require(isinstance(evidence,dict) and set(evidence)=={'version','state','reason','model','source','score_notice','thresholds','limitations','subjects','unassigned_detections'})
    require(type(evidence['version']) is int and evidence['version']==1 and evidence['state'] in ('complete','unavailable'))
    require(evidence['reason']==(COMPLETE if evidence['state']=='complete' else UNAVAILABLE))
    require(evidence['model']==_metadata() and evidence['score_notice']==SCORE_NOTICE and evidence['thresholds']==THRESHOLDS and evidence['limitations']==LIMITATIONS)
    source=evidence['source']
    require(isinstance(source,dict) and set(source)=={'width','height','preview_width','preview_height'} and all(type(v) is int and 0<v<=100000 for v in source.values()))
    require(isinstance(evidence['subjects'],list) and len(evidence['subjects'])==len(subjects))
    ids=[]
    def valid_candidate(d, extra):
        require(_valid_detection(d,source['width'],source['height']) and set(d)=={'plate_id','box','detector_score','ocr_score','text',extra})
        require(isinstance(d['plate_id'],str) and re.fullmatch(r'plate-([1-9]|1[0-9]|20)',d['plate_id']) is not None)
        ids.append(d['plate_id'])
    for i,(row,s) in enumerate(zip(evidence['subjects'],subjects)):
        require(isinstance(row,dict) and set(row)=={'id','status','candidates','manual_correction'} and row['id']==s['id'] and isinstance(row['candidates'],list))
        for d in row['candidates']:
            valid_candidate(d,'evidence_id')
            require(d['evidence_id']==s['evidence_ids'][-1] and _matches(d,subjects,source)==[i])
        expected='not_applicable' if s['kind']=='person' else ('unavailable' if evidence['state']=='unavailable' else _status(row['candidates']))
        require(row['status']==expected)
        if expected in ('not_applicable','unavailable'): require(not row['candidates'])
        correction=row['manual_correction']
        if correction is not None:
            require(isinstance(correction,dict) and set(correction)=={'text','source','plate_id'} and correction['source']=='manual'
                    and _text_valid(correction['text']) and bool(correction['text'].strip())
                    and correction['plate_id'] in [d['plate_id'] for d in row['candidates']])
    require(isinstance(evidence['unassigned_detections'],list))
    for d in evidence['unassigned_detections']:
        valid_candidate(d,'reason')
        matches=_matches(d,subjects,source)
        require(len(matches)!=1 and d['reason']==('ambiguous_vehicle_overlap' if matches else 'no_numbered_vehicle_match'))
    require(len(ids)<=20 and len(set(ids))==len(ids) and (evidence['state']=='complete' or not ids))
    return evidence


def correct_plate_evidence(evidence, subjects, vehicle_id, text, *, plate_id=None):
    validate_plate_evidence(evidence,subjects)
    if not _text_valid(text) or not text.strip():
        raise ValueError('Manual transcription must contain 1–16 uppercase letters, digits or spaces.')
    result=copy.deepcopy(evidence)
    row=next((r for r in result['subjects'] if r['id']==vehicle_id),None)
    if row is None or not row['candidates']:
        raise ValueError('Manual transcription requires a detected plate bound to this vehicle.')
    if plate_id is None and len(row['candidates'])==1: plate_id=row['candidates'][0]['plate_id']
    if plate_id not in [d['plate_id'] for d in row['candidates']]:
        raise ValueError('Choose a plate candidate belonging to this vehicle.')
    row['manual_correction']={'text':text,'source':'manual','plate_id':plate_id}
    return validate_plate_evidence(result,subjects)
