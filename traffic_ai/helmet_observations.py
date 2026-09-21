"""Single-frame model observations using existing unique rider/head association."""
from collections import Counter
from dataclasses import asdict
import cv2
from .geometry import containment
from .riders import associate_riders

NOTICE=('Model-assisted head observations with single-frame geometric motorcycle/person association. '
        'Unknown includes missing, weak, clipped, conflicting or shared head evidence. '
        'Missing helmet detections never establish a bare head. These are not legal verdicts or proof of riding.')
LABELS={'helmet_present':'helmet observed','bare_head':'bare head observed','unknown':'helmet unknown'}


def observe(detections,shape):
    bikes=[d for d in detections if d.label=='motorcycle']
    people=[d for d in detections if d.label=='person']
    heads=[d for d in detections if d.model=='iam-tsr/yolov8n-helmet-detection']
    associations=associate_riders(bikes,people,heads)
    indices={id(d):i for i,d in enumerate(detections)}
    h,w=shape[:2]
    def clipped(d):
        x,y,x2,y2=d.box
        return x<=2 or y<=2 or x2>=w-2 or y2>=h-2
    rows=[]
    for i,bike in enumerate(bikes):
        evidence=associations[i]
        for person in evidence.associated_people or [None]:
            head_area=(person.box[0],person.box[1],person.box[2],person.box[1]+.4*(person.box[3]-person.box[1])) if person else None
            candidates=[d for d in heads if head_area and containment(d.box,head_area)>=.8]
            supported=[d for d in evidence.bare_heads+evidence.helmet_heads if d in candidates]
            state='unknown';reason='No uniquely associated rider candidate'
            if person is not None:
                reason='Missing, weak, conflicting or shared head evidence'
                if not evidence.association_confident:
                    reason='Motorcycle/person association is ambiguous'
                elif min(bike.confidence,person.confidence)<.5:
                    reason='Weak motorcycle/person candidate'
                elif clipped(bike) or clipped(person) or any(clipped(d) for d in supported):
                    reason='Clipped motorcycle, person or head candidate'
                elif len({'helmet' if d.label=='helmet' else 'bare' for d in candidates
                          if d.confidence>=.4 and d.label in {'helmet','no_helmet','no-helmet'}})>1:
                    reason='Conflicting explicit helmet and bare-head candidates; including lower-confidence model evidence'
                elif any(min(d.box[2]-d.box[0],d.box[3]-d.box[1])<8 or
                         (d.box[2]-d.box[0])*(d.box[3]-d.box[1])<80 for d in supported):
                    reason='Head candidate has insufficient image resolution'
                elif len(supported)==1:
                    state='helmet_present' if supported[0].label=='helmet' else 'bare_head'
                    reason='Explicit high-confidence head class uniquely associated with this person and motorcycle'
            rows.append({'state':state,'label':LABELS[state],'reason':reason,
                'motorcycle_detection_index':indices[id(bike)],
                'person_detection_index':indices[id(person)] if person else None,
                'head_detection_indices':[indices[id(d)] for d in supported] if state!='unknown' else [],
                'candidate_head_detection_indices':[indices[id(d)] for d in candidates],
                'box_xyxy':list(person.box if person else bike.box),
                'association_basis':'Single-frame unique geometric association; shared riding motion unverified',
                'legal_violation':False,'eligible_for_rules':False})
    return {'schema_version':1,'notice':NOTICE,'observations':rows,'counts':dict(Counter(r['state'] for r in rows)),
            'detections':[asdict(d) for d in detections],
            'detection_indexing':'Zero-based indices into this record detections; no inferred missing heads.',
            'motorcycle_candidates':len(bikes),'unassociated_heads':sum(not any(indices[id(d)] in r['candidate_head_detection_indices'] for r in rows) for d in heads),
            'legal_violation':False,'eligible_for_rules':False}


def draw(rgb,summary):
    out=rgb.copy()
    for row in summary['observations']:
        x,y,x2,y2=map(round,row['box_xyxy']);colour=(60,190,235) if row['state']=='helmet_present' else (255,180,40) if row['state']=='bare_head' else (180,180,180)
        cv2.rectangle(out,(x,y),(x2,y2),colour,2)
        pos=(max(2,min(out.shape[1]-170,x)),max(18,y-5))
        cv2.putText(out,row['label'],pos,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,row['label'],pos,cv2.FONT_HERSHEY_SIMPLEX,.5,colour,1,cv2.LINE_AA)
    return out
