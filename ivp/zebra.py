"""Neutral, reviewed-band image overlap. This module makes no legal decisions."""
from collections import Counter
import hashlib
import json
import math

import cv2
import numpy as np

LABEL = 'on zebra crossing'
NOTICE = ('Image-space vehicle rectangle overlap with a user-reviewed full crossing band. '
          'A rectangle includes background and is not a measured physical footprint. '
          'This is an occupancy observation, never a legal violation or speed measurement.')
VEHICLES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'vehicle'}
POLICY = {'eligible_for_rules': False, 'eligible_for_geometry': False,
          'eligible_for_pixel_exclusion': False, 'legal_violation': False}


def validate_region(value, media_id, revision, reference_seconds=None):
    """Only an explicit current-source human review may define this band.

    Normalized vertices use display-oriented image extent (0..width/height).
    Paint-survey candidates, including nested legacy records, are not accepted.
    """
    if not isinstance(value, dict) or set(value) != {
        'schema_version', 'semantic', 'reviewed', 'media_id', 'source_revision',
        'reference_seconds', 'points'}:
        raise ValueError('Review a full zebra crossing band on the current frame first.')
    if (type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['semantic'] != 'zebra_crossing' or value['reviewed'] is not True
            or value['media_id'] != media_id or value['source_revision'] != revision):
        raise ValueError('Crossing review belongs to different or outdated footage.')
    stamp = value['reference_seconds']
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
            or not math.isfinite(stamp) or stamp < 0
            or (reference_seconds is not None and abs(stamp-reference_seconds) > 1e-6)):
        raise ValueError('Review the crossing band on the requested reference frame.')
    points = value['points']
    if (not isinstance(points, list) or len(points) != 4
            or any(not isinstance(p, list) or len(p) != 2 for p in points)
            or any(isinstance(v, bool) or not isinstance(v, (float, int))
                   or not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p)):
        raise ValueError('Crossing band needs four normalized corners.')
    polygon = np.asarray(points, np.float32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < .0005:
        raise ValueError('Choose a non-crossing convex band with useful area.')
    if np.any(polygon <= 0) or np.any(polygon >= 1):
        raise ValueError('The full crossing band must be inside the frame; a cropped band needs fresh review.')
    result = {**value, 'points': [[float(v) for v in p] for p in points]}
    result['region_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:20]
    return result


def assess(region, box, shape, *, label='vehicle', confidence=1., track_id=None,
           uncertain=None, timestamp=0., valid_mask=None):
    """Continuous polygon area, not visible paint pixels or a bottom-point test."""
    h, w = shape[:2]
    polygon = np.asarray(region['points'], np.float32) * (w, h)
    crossing_area = float(cv2.contourArea(polygon.astype(np.float32)))
    def number(value):
        return (not isinstance(value,(bool,np.bool_)) and isinstance(value,(int,float,np.integer,np.floating))
                and math.isfinite(value))
    safe_box = [float(v) if number(v) else None for v in box]
    safe_confidence = float(confidence) if number(confidence) else None
    row = {'label': None, 'status': 'withheld', 'reason': None, 'track_id': track_id,
           'timestamp_seconds': float(timestamp), 'vehicle_label': label,
           'confidence': safe_confidence, 'vehicle_box_xyxy': safe_box,
           'crossing_region_id': region['region_id'], 'overlap_area_px2': None,
           'vehicle_overlap_fraction': None, 'crossing_overlap_fraction': None,
           'basis': 'vehicle_rectangle_intersects_reviewed_full_band',
           'coordinate_space': 'display-oriented analysis image extent', **POLICY}
    if len(box) != 4 or not all(number(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        row['reason'] = 'invalid vehicle rectangle'
        return row
    x1, y1, x2, y2 = box
    if label not in VEHICLES:
        row['reason'] = 'not a vehicle candidate'
        return row
    if safe_confidence is None or confidence < .25 or confidence > 1:
        row['reason'] = 'weak vehicle candidate'
        return row
    if uncertain:
        row['reason'] = str(uncertain)
        return row
    if x1 <= 2 or y1 <= 2 or x2 >= w-2 or y2 >= h-2:
        row['reason'] = 'vehicle clipped by frame'
        return row
    if valid_mask is not None:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if any(not valid_mask[min(h-1, int(y)), min(w-1, int(x))] for x, y in corners):
            row['reason'] = 'vehicle touches a camera-warp border'
            return row
    rectangle = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    area, intersection = cv2.intersectConvexConvex(polygon.astype(np.float32), rectangle)
    area = min(crossing_area,(x2-x1)*(y2-y1),max(0., float(area)))
    fraction = area / ((x2-x1)*(y2-y1))
    row.update(overlap_area_px2=area, vehicle_overlap_fraction=fraction,
               crossing_overlap_fraction=area / crossing_area)
    extent = np.ptp(intersection.reshape(-1,2),axis=0) if intersection is not None else (0,0)
    if area <= 1e-6:
        row.update(status='no_overlap', reason='vehicle rectangle does not overlap the reviewed band')
    elif area < 16 or fraction < .15 or min(extent) < 2:
        row['reason'] = 'small or grazing overlap; review required'
    else:
        row.update(status='observed', label=LABEL,
                   reason='vehicle rectangle overlaps the reviewed full crossing band')
    return row


class Occupancy:
    """Bounded per-track episodes. Only consecutive current observations confirm.

    Confirmation: >=3 observations spanning >=0.2s. A >=0.5s missing gap
    or registration failure clears support, retaining an established episode until
    two current no-overlap observations. Equal timestamps never add support.
    Geometry is invalidated on a camera/scene discontinuity until human review.
    """
    def __init__(self, region):
        self.region = region
        self.active = {}
        self.last_stamp = None
        self.invalidated = False
        self.episode_count = 0
        self.counts = Counter()
        self.state_evictions = 0

    def step(self, tracks, stamp, shape, *, registration_valid=True, scene_changed=False, valid_mask=None):
        duplicate = self.last_stamp is not None and stamp == self.last_stamp
        if scene_changed or (self.last_stamp is not None and stamp < self.last_stamp):
            self.invalidated = True
        if (not registration_valid or self.invalidated
                or self.last_stamp is not None and stamp-self.last_stamp >= .5):
            for entry in self.active.values():
                entry.update(hits=0,start=stamp,clear_hits=0)
        self.last_stamp = stamp
        seen, rows = set(), []
        for key, t in tracks:
            if key == 'SELECTED':
                continue  # Not an automatic current vehicle detection.
            x, y, bw, bh = t['box']
            uncertainty = ('crossing review invalidated by camera/scene discontinuity' if self.invalidated
                           else 'duplicate frame timestamp' if duplicate else
                           'registration uncertain' if not registration_valid else
                           t.get('speed_blocked') or
                           (None if t.get('tracking_state') == 'observed' else 'track not currently unambiguously observed'))
            row = assess(self.region, (x,y,x+bw,y+bh), shape,
                         label=t.get('model_label','vehicle'), confidence=t.get('confidence',0),
                         track_id=key, uncertain=uncertainty, timestamp=stamp, valid_mask=valid_mask)
            seen.add(key)
            if row['status'] == 'observed':
                entry = self.active.get(key)
                if entry is None:
                    entry = {'start': stamp, 'last': stamp, 'hits': 0, 'clear_hits':0, 'episode_id': None}
                    self.active[key] = entry
                entry['hits'] += 1
                entry['last'] = stamp
                entry['clear_hits'] = 0
                if entry['hits'] >= 3 and stamp-entry['start'] >= .2-1e-9:
                    row['new_episode'] = entry['episode_id'] is None
                    if entry['episode_id'] is None:
                        self.episode_count += 1
                        entry['episode_id'] = self.episode_count
                    row['episode_id'] = entry['episode_id']
                    row['episode_first_overlap_seconds'] = entry['start']
                else:
                    row.update(status='pending', label=None, reason='waiting for repeated overlap observations')
            else:
                entry = self.active.get(key)
                if entry is not None and not duplicate:
                    entry.update(hits=0,start=stamp,last=stamp)
                    entry['clear_hits'] = entry['clear_hits']+1 if row['status']=='no_overlap' else 0
                    if entry['clear_hits'] >= 2:
                        self.active.pop(key)
            self.counts[row['status']] += 1
            rows.append(row)
        # Missing/occluded frames cannot count as repeated observations. Retain
        # confirmed episode IDs only through a brief gap; restart confirmation.
        for key in list(self.active):
            if key not in seen:
                if self.active[key]['episode_id'] is None and stamp-self.active[key]['last'] >= .5:
                    del self.active[key]
                else:
                    self.active[key]['hits'] = 0
                    self.active[key]['start'] = stamp
                    self.active[key]['clear_hits'] = 0
        # Compact per-identity episode memory is bounded, independent of video
        # duration. Very old evicted identities require fresh confirmation.
        while len(self.active)>1024:
            key=min(self.active,key=lambda k:self.active[k]['last'])
            del self.active[key]
            self.state_evictions += 1
        return rows

    def summary(self):
        return {'schema_version': 1, 'label': LABEL, 'notice': NOTICE, 'reviewed_region': self.region,
                'episodes': self.episode_count, 'sample_counts': dict(self.counts),
                'review_invalidated': self.invalidated,
                'confirmation': '3 distinct current observations spanning 0.2 seconds; 0.5 second gap resets support; two observed exits end an episode; 1024 identity memory bound',
                'identity_state_evictions':self.state_evictions,
                'thresholds': {'minimum_overlap_px2': 16, 'minimum_vehicle_overlap_fraction': .15,
                               'minimum_intersection_extent_px':2},
                **POLICY}


def draw(rgb, region, rows):
    out = rgb.copy()
    h, w = out.shape[:2]
    points = np.rint(np.asarray(region['points'])*(w,h)).astype(np.int32)
    cv2.polylines(out, [points], True, (255,180,40), 2, cv2.LINE_AA)
    for row in rows:
        if row['status'] != 'observed':
            continue
        x1,y1,x2,y2 = map(round,row['vehicle_box_xyxy'])
        cv2.rectangle(out,(x1,y1),(x2,y2),(255,180,40),2)
        pos = (max(2,min(w-170,x1)),max(18,y1-7))
        cv2.putText(out,LABEL,pos,cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(out,LABEL,pos,cv2.FONT_HERSHEY_SIMPLEX,.55,(255,180,40),1,cv2.LINE_AA)
    return out
