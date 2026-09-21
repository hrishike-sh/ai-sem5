from copy import deepcopy
from math import dist

from .geometry import crossed_segment, point_in_polygon, signed_distance


class TurnContexts:
    """Retain an observed entry's sign applicability while its vehicle exits."""

    def __init__(self, lifetime=8.):
        self.lifetime = lifetime
        self.tracks = {}

    def reset(self):
        self.tracks.clear()

    def observe(self, vehicle, timestamp, scene):
        self.tracks = {key: value for key, value in self.tracks.items()
                       if 0 <= timestamp - value['last_seen'] <= .75}
        identity = vehicle.track_id
        if identity is None:
            return None
        state = self.tracks.get(identity)
        if not scene.stationary_camera or scene.confidence < .8:
            self.tracks.pop(identity, None)
            return None
        point = vehicle.ground_point
        if state and (state['scene_id'] != scene.scene_id or
                      dist(point, state['point']) > 5 * max(8., vehicle.box[2] - vehicle.box[0])):
            self.tracks.pop(identity, None)
            state = None
        if state and timestamp <= state['last_seen']:
            raise ValueError('Turn-context timestamps must increase')
        if state and timestamp - state['started_at'] > self.lifetime:
            self.tracks.pop(identity, None)
            state = None
        if state and state.get('entry'):
            state.update(last_seen=timestamp, point=point)
            return deepcopy(state['entry'])
        approach = scene.source_evidence.get('approach')
        sign = scene.source_evidence.get('applicable_sign')
        applicable = (approach and sign and scene.stop_line and scene.prohibited_turns
                      and sign.get('associated_approach') == approach['id']
                      and point_in_polygon(point, approach['polygon']))
        if not applicable:
            self.tracks.pop(identity, None)
            return None
        signature = (approach['id'], repr(scene.stop_line), repr(sign))
        if state and state['signature'] != signature:
            state = None
        if state and crossed_segment(state['anchor'], point, scene.stop_line, margin=3):
            entry = {'track_id': identity, 'scene_id': scene.scene_id,
                     'entry_timestamp': timestamp, 'approach_id': approach['id'],
                     'entry_from': state['anchor'], 'entry_to': point,
                     'prohibited_turns': sorted(scene.prohibited_turns), 'sign': deepcopy(sign)}
            state.update(entry=entry, last_seen=timestamp, point=point)
            return deepcopy(entry)
        if signed_distance(point, scene.stop_line) < -3:
            self.tracks[identity] = {'signature': signature, 'scene_id': scene.scene_id,
                                     'anchor': point, 'point': point, 'last_seen': timestamp,
                                     'started_at': timestamp, 'entry': None}
        elif state:
            state.update(last_seen=timestamp, point=point)
        return None
