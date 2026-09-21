"""Classical motion measurements for fixed-camera video. No model weights."""
from collections import deque
import math

import cv2
import numpy as np
from .speed import SpeedEstimator
from .stabilization import TranslationStabilizer
from .target_track import SelectedTarget
from .auto_track import AutomaticTracker
from .speed_display import SpeedDisplay


class RegionTracker:
    """Short-lived region identities, not semantic vehicle identities."""
    def __init__(self, axis='horizontal', position=.65, min_age=0):
        self.axis, self.position = axis, position
        self.min_age = min_age
        self.tracks, self.next_id = {}, 1
        self.counts = {'positive': 0, 'negative': 0}
        self.ambiguous_boxes = 0

    def update(self, boxes, stamp, shape):
        h, w = shape
        coordinate = 1 if self.axis == 'horizontal' else 0
        line = self.position * (h if coordinate else w)
        gate = max(25, min(w, h)*.12)
        self.tracks = {k: t for k, t in self.tracks.items() if stamp-t['time'] <= .5}
        # Do not attach a merged foreground blob to just one of its old tracks.
        # Keep identities briefly, but discard velocity/speed history across the gap.
        blocked = set()
        for index, (x,y,bw,bh) in enumerate(boxes):
            contained = []
            for key, track in self.tracks.items():
                predicted = track['centre'] + track['velocity']*min(stamp-track['time'], .2)
                if track['hits'] >= 3 and x <= predicted[0] <= x+bw and y <= predicted[1] <= y+bh:
                    contained.append(key)
            if len(contained) > 1:
                blocked.add(index)
                for key in contained:
                    self.tracks[key].pop('speed_history', None)
                    self.tracks[key]['side'] = 0
        self.ambiguous_boxes = len(blocked)
        pairs = []
        for key, track in self.tracks.items():
            dt = max(0, stamp-track['time'])
            predicted = track['centre'] + track['velocity']*min(dt, .2)
            for index, (x, y, bw, bh) in enumerate(boxes):
                if index in blocked:
                    continue
                centre = np.array([x+bw/2, y+bh/2])
                distance = np.linalg.norm(predicted-centre)
                area_ratio = bw*bh/max(1, track['box'][2]*track['box'][3])
                if distance <= gate and .5 <= area_ratio <= 2:
                    pairs.append((distance + gate*.2*abs(math.log(area_ratio)), key, index))
        used_tracks, used_boxes = set(), set()
        matches = {}
        for _, key, index in sorted(pairs):
            if key not in used_tracks and index not in used_boxes:
                matches[index] = key
                used_tracks.add(key); used_boxes.add(index)
        visible, events = [], []
        for index, box in enumerate(boxes):
            if index in blocked:
                continue
            x, y, bw, bh = box
            centre = np.array([x+bw/2, y+bh/2])
            key = matches.get(index)
            if key is None:
                key = self.next_id; self.next_id += 1
                self.tracks[key] = {'centre': centre, 'velocity': np.zeros(2), 'time': stamp,
                                    'hits': 0, 'trail': deque(maxlen=40), 'side': 0, 'counted': False,
                                    'box': box, 'start': stamp}
            t = self.tracks[key]
            if stamp > t['time']:
                t['velocity'] = .5*t['velocity'] + .5*(centre-t['centre'])/(stamp-t['time'])
            t.update(centre=centre, time=stamp, box=box, hits=t['hits']+1)
            t['trail'].append(tuple(map(int, centre)))
            side = 1 if centre[coordinate] > line+5 else -1 if centre[coordinate] < line-5 else 0
            if t['hits'] >= 3 and stamp-t['start'] >= self.min_age:
                if side and t['side'] and side != t['side'] and not t['counted']:
                    direction = 'positive' if side > 0 else 'negative'
                    self.counts[direction] += 1; t['counted'] = True
                    events.append({'timestamp_seconds': stamp, 'track_id': key, 'direction': direction,
                                   'x': float(centre[0]), 'y': float(centre[1])})
                if side:
                    t['side'] = side
                visible.append((key, t))
        return visible, events


class MotionAnalyzer:
    def __init__(self, axis='horizontal', position=.65, rectangle=None, calibration=None, target_rectangle=None, tracking_mode='ivp', output_view='tracks', frame_interval=1/30, speed_display=None):
        if axis not in ('horizontal', 'vertical') or isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position) or not .1 <= position <= .9:
            raise ValueError('Counting line must be horizontal/vertical and between 10% and 90%.')
        rectangle = [0, 0, 1, 1] if rectangle is None else rectangle
        if (not isinstance(rectangle, list) or len(rectangle) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in rectangle)
            or rectangle[2]-rectangle[0] < .05 or rectangle[3]-rectangle[1] < .05):
            raise ValueError('Choose a road region at least 5% wide and high.')
        # A line outside the region cannot produce useful crossings.
        index = 1 if axis == 'horizontal' else 0
        if not rectangle[index] < position < rectangle[index+2]:
            raise ValueError('Place the counting line inside the selected road region.')
        if tracking_mode not in ('ivp','vehicles') or output_view not in ('tracks','stages'):
            raise ValueError('Choose IVP or vehicle tracking and a tracks or stages output view.')
        self.tracking_mode,self.output_view=tracking_mode,output_view
        self.speed_display = SpeedDisplay(speed_display, calibration, tracking_mode) if speed_display is not None else None
        self.axis, self.position, self.rectangle = axis, position, rectangle
        if tracking_mode=='vehicles':
            from traffic_ai.vehicle_motion import VehicleTracker
            self.tracker=VehicleTracker(axis,position,min_age=.3,frame_interval=frame_interval)
            self.tracker.rectangle=rectangle
        else:
            self.tracker = AutomaticTracker(axis, position, min_age=.3,frame_interval=frame_interval)
        self.speed = SpeedEstimator(calibration,frame_interval=frame_interval)
        self.stabilizer = TranslationStabilizer(rectangle)
        self.target = SelectedTarget(target_rectangle,frame_interval=frame_interval) if target_rectangle is not None else None
        self.target_speed = SpeedEstimator(calibration,frame_interval=frame_interval)
        self.background = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
        self.previous = None
        self.heat = None
        self.last_stamp = None
        self.warm_until = .7
        self.valid_seconds = 0.
        self.suppressed = 0
        self.ambiguous_frames = 0
        self.frames = 0
        self.peak = None
        self.peak_activity = -1
        self.last_rgb = None
        self.flow_points = []
        self.flow_samples = 0
        self.flow_sum = 0.

    def step(self, rgb, stamp, scene_change=False):
        if scene_change:
            self.stabilizer = TranslationStabilizer(self.rectangle)
        rgb, registered = self.stabilizer.step(rgb)
        # Neutral crossing occupancy needs an actual current registration result,
        # not merely the absence of a legacy failure flag when no texture exists.
        self.registration_observed = self.stabilizer.frames == 1 or registered
        h, w = rgb.shape[:2]
        if self.speed_display:
            self.speed_display.frame_size = [w, h]
        if self.heat is None:
            self.heat = np.zeros((h, w), np.float32)
        if self.heat.shape != (h, w):
            raise ValueError('Video dimensions changed.')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        shift = 0.
        coherent = False
        flow_pairs = []
        # Sparse Lucas–Kanade flow tests whether most tracked texture moved
        # together. This is a rejection guard, not camera stabilization.
        if self.previous is not None:
            corners = cv2.goodFeaturesToTrack(self.previous, 500, .015, 8)
            if corners is not None and len(corners) >= 12:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, corners, None)
                if moved is not None:
                    back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None)
                    reliable = (status.ravel() == 1) & (back_status.ravel() == 1) & (np.linalg.norm(back-corners,axis=2).ravel() < 1.)
                    starts, ends = corners.reshape(-1,2)[reliable], moved.reshape(-1,2)[reliable]
                    vectors = ends-starts
                    flow_pairs = list(zip(starts,ends))
                    if len(vectors) >= 12:
                        median = np.median(vectors, axis=0)
                        shift = float(np.linalg.norm(median))
                        coherent = np.mean(np.linalg.norm(vectors-median, axis=1) < 2) > .65
        self.previous = gray
        # Verified static-context registration takes precedence over the median
        # foreground flow: a crowd moving together is not camera movement.
        camera_change = scene_change or (shift > 2.5 and coherent and not registered)
        if camera_change:
            self.speed.invalidate()
            self.target_speed.invalidate()
            self.background = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
            if hasattr(self.tracker,'reset'):self.tracker.reset()
            else:self.tracker.tracks.clear()
            self.warm_until = stamp+.7
        raw = self.background.apply(cv2.GaussianBlur(rgb, (5, 5), 0), learningRate=.03 if stamp < self.warm_until else .003)
        mask = ((raw == 255)*255).astype(np.uint8)  # MOG2's 127 shadow label is excluded.
        roi = np.zeros_like(mask)
        x1, y1, x2, y2 = self.rectangle
        bounds = int(x1*w), int(y1*h), math.ceil(x2*w), math.ceil(y2*h)
        left, top, right, bottom = bounds
        roi[top:bottom, left:right] = 255
        roi &= self.stabilizer.valid_mask
        mask &= roi
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        mask &= roi
        fraction = np.count_nonzero(mask)/max(1, np.count_nonzero(roi))
        reason = 'registration uncertain' if self.stabilizer.failed else 'camera/scene change' if camera_change else 'background warm-up' if stamp < self.warm_until else 'widespread foreground' if fraction > .5 else 'measured'
        valid = reason == 'measured'
        boxes = []
        min_area = max(35, round(h*w*.00025))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        keep = np.zeros(count, np.uint8)
        ambiguous = 0
        if valid:
            for i in range(1, count):
                x, y, bw, bh, area = map(int, stats[i])
                if min_area <= area <= h*w*.2 and bw >= 5 and bh >= 5:
                    keep[i] = 255
                    if bw*bh > h*w*.15:
                        ambiguous += 1
                    else:
                        boxes.append((x, y, bw, bh))
        mask = keep[labels]
        tracking_valid=valid or (self.tracking_mode=='vehicles' and not self.stabilizer.failed and not camera_change)
        if tracking_valid:
            self.tracker.valid_mask=self.stabilizer.valid_mask
            tracks, events = self.tracker.update(boxes, stamp, (h, w), gray, mask, rgb)
            ambiguous += self.tracker.ambiguous_boxes
            for _, track in tracks:
                self.speed.update(track, stamp, (h,w))
        else:
            if self.tracking_mode=='ivp':
                self.tracker.tracks.clear()
                self.tracker.update([], stamp, (h,w), gray, mask, rgb)
            tracks, events = [], []
        if not valid:self.suppressed += 1
        if ambiguous:
            self.ambiguous_frames += 1
        if self.target:
            target=self.target.step(rgb,stamp,not self.stabilizer.failed and not camera_change)
            if target is not None:
                self.target_speed.update(target,stamp,(h,w))
                tracks.append(('SELECTED',target))
        dt = 0 if self.last_stamp is None else stamp-self.last_stamp
        if valid and dt > 0:
            self.heat += (mask > 0)*dt
            self.valid_seconds += dt
        self.last_stamp = stamp; self.frames += 1; self.last_rgb = rgb.copy()
        activity = float(np.count_nonzero(mask)/max(1, np.count_nonzero(roi))*100) if valid else None
        annotated = rgb.copy()
        # Texture motion remains visible when touching vehicles merge into blobs.
        # These arrows are local optical-flow measurements, not object identities.
        flow_speeds = []
        if valid and dt > 0:
            for p,q in flow_pairs:
                qx,qy = map(int,q)
                if 0 <= qx < w and 0 <= qy < h and mask[qy,qx] and np.linalg.norm(q-p) > .3:
                    flow_speeds.append(float(np.linalg.norm(q-p)/dt))
                    end = q+(q-p)*3
                    cv2.arrowedLine(annotated, tuple(map(int,p)),tuple(map(int,end)),(110,235,100),1,tipLength=.3)
        flow_speed = float(np.median(flow_speeds)) if len(flow_speeds) >= 5 else None
        if flow_speed is not None:
            self.flow_samples += 1; self.flow_sum += flow_speed
        if self.speed.calibration:
            points = (self.speed.calibration.points*[w,h]).astype(np.int32)
            cv2.polylines(annotated,[points],True,(200,140,255),2)
        flow_view=annotated.copy()
        # Keep the main result legible; the arrow field has its own IVP stage.
        annotated=rgb.copy()
        if self.speed.calibration:
            cv2.polylines(annotated,[points],True,(200,140,255),2)
        cv2.rectangle(annotated, (left, top), (right-1, bottom-1), (170, 180, 190), 1)
        if self.axis == 'horizontal':
            a,b=(left,round(h*self.position)),(right-1,round(h*self.position))
        else:
            a,b=(round(w*self.position),top),(round(w*self.position),bottom-1)
        cv2.line(annotated,a,b,(255,180,40),2)
        display_note = (f" | display > {self.speed_display.config['limit']} {self.speed_display.config['unit']}"
                        if self.speed_display and self.speed_display.config['enabled'] else '')
        for key,t in tracks:
            if self.speed_display and not self.speed_display.observe(t, (h, w)):
                continue  # Rendering only: tracks, speed history, events and exports are retained.
            x,y,bw,bh=map(int,t['box'])
            color=(30,220,220) if key=='SELECTED' else tuple(int(v) for v in cv2.cvtColor(np.uint8([[[(int(key)*37)%180,160,245]]]),cv2.COLOR_HSV2RGB)[0,0])
            cv2.rectangle(annotated,(x,y),(x+bw,y+bh),color,2)
            label='SELECTED' if key=='SELECTED' else f"{'V' if self.tracking_mode=='vehicles' else 'M'}{key}"
            if t.get('tracking_state')=='ambiguous overlap':label+=' ?'
            lines=[label]
            if t.get('speed') is not None:lines.append(f"{t['speed']:.2f} {t['speed_unit']}" if self.speed_display and self.speed_display.config['enabled'] else f"{t['speed']:.1f} {t['speed_unit']}")
            tx=max(0,min(w-90,x));ty=max(15,y)
            for line in lines:
                width=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.37,1)[0][0]
                cv2.rectangle(annotated,(tx,ty-12),(min(w-1,tx+width+6),ty+3),(18,30,34),-1)
                cv2.putText(annotated,line,(tx+3,ty),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1,cv2.LINE_AA)
                ty+=15
            if len(t['trail']) > 1:
                cv2.polylines(annotated,[np.array(t['trail'],np.int32)],False,color,1,cv2.LINE_AA)
        metrics={'timestamp_seconds':stamp,'state':reason,'moving_area_percent':activity,
                 'active_tracks':len(tracks),'crossings_positive':self.tracker.counts['positive'],
                 'crossings_negative':self.tracker.counts['negative'],'global_flow_px':shift,
                 'unresolved_large_regions':ambiguous,'flow_speed_px_s':flow_speed,
                 'flow_points':len(flow_speeds),'speed_estimates':sum(t.get('speed') is not None for _,t in tracks),
                 'stabilization_x_px':float(self.stabilizer.offset[0]),'stabilization_y_px':float(self.stabilizer.offset[1])}
        metrics.update({f'camera_m{i}':float(v) for i,v in enumerate(self.stabilizer.matrix.ravel())})
        new_peak = valid and activity > self.peak_activity
        # Full-size tracks need stage images only when saving a new peak.
        if self.output_view == 'stages' or new_peak:
            heat=self.heatmap()
            mosaic=np.zeros_like(rgb)
            half_w,half_h=w//2,h//2
            panels=[(rgb,'STABILIZED SOURCE'),(annotated,'TRACKS + SPEED'+display_note),
                    (cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB),'CLEANED FOREGROUND'),(heat,'RELATIVE MOTION HEATMAP')]
            for idx,(pixels,title) in enumerate(panels):
                x,y=(idx%2)*half_w,(idx//2)*half_h
                mosaic[y:y+half_h,x:x+half_w]=cv2.resize(pixels,(half_w,half_h))
                cv2.rectangle(mosaic,(x,y),(x+half_w,y+23),(19,30,36),-1)
                font_scale=min(.38,(half_w-14)/max(1,cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,1,1)[0][0]))
                cv2.putText(mosaic,title,(x+7,y+16),cv2.FONT_HERSHEY_SIMPLEX,font_scale,(235,242,242),1)
            total=sum(self.tracker.counts.values())
            cv2.rectangle(mosaic,(0,h-24),(w,h),(19,30,36),-1)
            count_label='candidate crossings' if self.tracking_mode=='vehicles' else 'region crossings'
            cv2.putText(mosaic,f'{stamp:5.2f}s | {reason} | {count_label}: {total}',
                        (8,h-8),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
        if new_peak:
            self.peak_activity=activity
            background = self.background.getBackgroundImage()
            self.peak={'mosaic':mosaic.copy(),'mask':mask.copy(),'tracks':annotated.copy(),'original':rgb.copy(),
                       'flow':flow_view,
                       'background':background.copy() if background is not None else rgb.copy(),
                       'raw-foreground':raw.copy(),'shadows':((raw==127)*255).astype(np.uint8),'timestamp':stamp}
        cv2.rectangle(annotated,(0,h-25),(w,h),(19,30,36),-1)
        mode_label='LOCAL VEHICLE DETECTOR + IVP' if self.tracking_mode=='vehicles' else 'AUTOMATIC IVP OBJECTS'
        footer=f'{mode_label} | {stamp:.2f}s | {len(tracks)} tracks | speed: {self.speed.summary()["unit"]}'+display_note
        footer_scale=min(.44,(w-16)/max(1,cv2.getTextSize(footer,cv2.FONT_HERSHEY_SIMPLEX,1,1)[0][0])) if display_note else .44
        cv2.putText(annotated,footer,(8,h-8),cv2.FONT_HERSHEY_SIMPLEX,footer_scale,(240,240,240),1)
        return annotated if self.output_view=='tracks' else mosaic, metrics, events, tracks

    def heatmap(self):
        maximum=float(self.heat.max())
        scale=(self.heat/maximum*255).astype(np.uint8) if maximum>0 else np.zeros_like(self.heat,np.uint8)
        color=cv2.cvtColor(cv2.applyColorMap(scale,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        output=self.last_rgb.copy() if self.last_rgb is not None else np.zeros((*self.heat.shape,3),np.uint8)
        active=scale>0
        # Exact integer equivalent of the previous quarter-weighted blend.
        blend=((output.astype(np.uint16)+3*color.astype(np.uint16))//4).astype(np.uint8)
        np.copyto(output,blend,where=active[:,:,None])
        return output

    def summary(self):
        return {'method':('Local YOLO11s vehicle proposals + bottom-centre-aware ByteTrack with repeated-observation recovery + IVP camera registration, foreground, optical flow and timestamp speed' if self.tracking_mode=='vehicles' else 'MOG2 discovery + morphology + automatic Shi-Tomasi / forward-backward Lucas-Kanade texture tracking + conservative occlusion recovery + timestamp speed'),
                **({'speed_display': self.speed_display.summary()} if self.speed_display else {}),
                'speed':self.speed.summary(),
                'tracking':self.tracker.summary(),
                'selected_target':{**self.target.summary(),'speed':self.target_speed.summary()} if self.target else None,
                'stabilization':self.stabilizer.summary(),
                'optical_flow':{'measured_frames':self.flow_samples,'mean_frame_median_px_s':self.flow_sum/self.flow_samples if self.flow_samples else None,
                                'meaning':'Moving foreground texture speed in analysis pixels/second; not vehicle speed.'},
                'uses_ai':self.tracking_mode=='vehicles','output_view':self.output_view,'frames':self.frames,'suppressed_frames':self.suppressed,
                'ambiguous_frames':self.ambiguous_frames,
                'measured_seconds':self.valid_seconds,'crossings':self.tracker.counts,
                'line':{'axis':self.axis,'position':self.position},'roi':self.rectangle,
                'parameters':{'gaussian_kernel':5,'mog2_history':300,'mog2_variance_threshold':25,
                              'warmup_seconds':.7,'learning_rate_warmup':.03,'learning_rate':.003,
                              'shadow_label_excluded':127,'opening_kernel':3,'closing_kernel':5,
                              'min_area_rule':'max(35, round(frame_area * 0.00025))',
                              'large_bbox_tracking_exclusion_fraction':.15,'track_confirm_hits':4,
                              'track_min_age_seconds':.3,'association_area_ratio':[.5,2],
                              'optical_flow_forward_backward_error_px':1.,'speed_window_seconds':[.5,self.speed.timing.speed_window],
                              'track_expiry_seconds':.65,'line_hysteresis_px':5,
                              'camera_flow_threshold_px':2.5,'camera_flow_coherence':.65},
                'peak_activity_percent':max(0,self.peak_activity),
                'peak_timestamp':self.peak['timestamp'] if self.peak else None,
                'heatmap_max_seconds':float(self.heat.max()) if self.heat is not None else 0,
                'limitations':[('Vehicle candidates come from the local detector; their identities and counts are not independently validated. Auto-rickshaw types are not classified.' if self.tracking_mode=='vehicles' else 'Counts refer to moving-region tracks, not independently validated vehicle counts.'),
                ('Overlaps, small vehicles and changing viewpoints can cause missed detections and identity switches.' if self.tracking_mode=='vehicles' else 'Touching vehicles can merge; one object can split into multiple regions. Occlusion can change identities.'),
                'Requires a near-fixed camera. Uncertain camera registration and scene changes suppress tracking; widespread foreground suppresses foreground measurements.',
                'Initial 0.7 seconds and resets warm up the background model. This excludes heatmap measurements but vehicle-mode detection can run during warm-up.',
                'Speed is estimated from stable region trajectories; segmentation jitter and identity changes can cause error. It is not enforcement-grade.',
                'km/h requires user-measured flat-road calibration. Outside that patch, speed is withheld. Without calibration only analysis pixels/second are reported.',
                'A question mark marks overlapping vehicle hypotheses. Their speed and line crossings are withheld until the ambiguity clears; IDs are not automatically merged.',
                'Short-gap ID recovery requires new observed detections. A recovered weak track starts a fresh speed window and never supplies a line crossing across the gap.',
                'No helmet status or violation decision is inferred. The IVP-only mode does not classify objects.']}
