"""Stream incident evidence, retaining only a bounded review index in memory."""
import csv
import json
import cv2
import numpy as np
from .media import png_bytes, write_json


def draw_geometry(image, config):
    h,w = image.shape[:2]
    if config.get('line'):
        p = np.array(config['line'])*[w,h]
        cv2.line(image,tuple(p[0].astype(int)),tuple(p[1].astype(int)),(250,120,60),2)
    for name,color in [('zone',(160,120,250)),('approach_zone',(80,210,210))]:
        if config.get(name):
            x,y,r,b = (np.array(config[name])*[w,h,w,h]).astype(int)
            cv2.rectangle(image,(x,y),(r,b),color,2)
    if config.get('synthetic'):
        cv2.putText(image,'SYNTHETIC - declared scale', (12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,210,70),2,cv2.LINE_AA)
    return image


class IncidentEvidence:
    def __init__(self, run, analyzer):
        self.run, self.analyzer = run, analyzer
        self.examples = []
        self.thumbnails = 0
        self.event_file = self.csv_file = None

    def __enter__(self):
        if self.analyzer.enabled:
            self.event_file = (self.run/'motion-incidents.jsonl').open('w',encoding='utf-8')
            self.csv_file = (self.run/'motion-incidents.csv').open('w',encoding='utf-8',newline='')
            self.writer = csv.DictWriter(self.csv_file,fieldnames=['event_id','timestamp_seconds','track_id','type','status','reason','synthetic','speed','speed_unit','threshold_kmh','measurement_guard_kmh','thumbnail'])
            self.writer.writeheader()
        return self

    def record(self, events, rgb, camera_matrix, transform_to_source, frame_number, source_revision):
        if not self.analyzer.enabled:
            return
        matrix = np.eye(3);matrix[:2,:] = camera_matrix
        to_source = np.asarray(transform_to_source) @ np.linalg.inv(matrix)
        h,w = rgb.shape[:2]
        for event in events:
            coords = event['coordinates']
            point = np.array([coords['anchor_normalized'][0]*w,coords['anchor_normalized'][1]*h,1.])
            mapped = to_source @ point
            coords.update(anchor_source_px=(mapped[:2]/mapped[2]).tolist(), analysis_to_source=to_source.tolist(),
                          source_frame='display-oriented original frame', analysis_size=[w,h])
            event['evidence_frame_number'] = frame_number
            event['evidence_timestamp_seconds'] = event['measurements'].get('observation_timestamp_seconds',event['timestamp_seconds'])
            event['source_revision'] = source_revision
            if self.thumbnails < 40:
                name = event['event_id']+'.png'
                image = draw_geometry(rgb.copy(),self.analyzer.config)
                x,y,bw,bh = map(int,coords['bbox_analysis_xywh'])
                cv2.rectangle(image,(x,y),(x+bw,y+bh),(255,210,70),2)
                label = f"{event['event_id']} | {event['type']} | {event['status']}"
                cv2.rectangle(image,(0,h-29),(w,h),(18,25,30),-1)
                cv2.putText(image,label,(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
                (self.run/name).write_bytes(png_bytes(image))
                event['thumbnail'] = name
                self.thumbnails += 1
            self.event_file.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            row = {k:event.get(k) for k in self.writer.fieldnames}
            row.update({k:event['measurements'].get(k) for k in ('speed','speed_unit','threshold_kmh','measurement_guard_kmh')})
            self.writer.writerow(row)
            if len(self.examples)<100:
                self.examples.append(event)

    def finish(self):
        summary = self.analyzer.summary()
        summary.update(events=self.examples, indexed_events=len(self.examples), thumbnails=self.thumbnails,
                       omitted_from_index=max(0,self.analyzer.serial-len(self.examples)),
                       thumbnail_policy='First 40 assessments; full event stream is in JSONL/CSV. Index shows first 100.')
        if self.event_file:
            self.event_file.flush();self.csv_file.flush()
            write_json(self.run/'motion-incidents.json',summary)
        return summary

    def __exit__(self,*args):
        if self.event_file:self.event_file.close()
        if self.csv_file:self.csv_file.close()
