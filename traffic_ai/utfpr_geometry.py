"""UTFPR author-calibration arithmetic, without models or vehicle truth access.

This module does not validate a feature's physical identity from pixels. Only a
separately justified observed road-contact point may use its planar projection.
Bounding-box bottoms and plate/body features are explicitly withheld. The app's
ordinary calibration and speed gates are not changed by this reference module.
"""
from fractions import Fraction
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from ivp.timing import FrameTiming

PUBLISHED_FPS = Fraction(603, 20)
FULL_SIZE = (1920, 1080)
MAX_COMPLETE_FRAMES = 200
RECTIFIED_PIXELS_PER_METRE = 60.0


def author_time(frame_index):
    if type(frame_index) is not int or not 0 <= frame_index < MAX_COMPLETE_FRAMES:
        raise ValueError('Only original complete frame indices0..199 are eligible.')
    return float(frame_index/PUBLISHED_FPS)


def timing_gate(frame_indices):
    times = [author_time(index) for index in frame_indices]
    if any(b <= a for a, b in zip(times, times[1:])):
        return {'state': 'withheld', 'reason': 'non-increasing or duplicate frame observations'}
    if len(times) < 8 or times[-1]-times[0] < .5:
        return {'state': 'withheld', 'reason': 'need at least8 observed positions spanning0.5seconds'}
    timing = FrameTiming(float(1/PUBLISHED_FPS))
    if any(b-a > timing.discontinuity for a, b in zip(times, times[1:])):
        return {'state': 'withheld', 'reason': 'observation gap; do not bridge missing positions'}
    return {'state': 'eligible_timing', 'observations': len(times), 'span_seconds': times[-1]-times[0],
            'timestamp_basis': 'author nominal frame index /30.15; not independently measured capture PTS'}


class AuthorPlanarCalibration:
    def __init__(self, matrices, lanes, measures):
        matrices = np.asarray(matrices, dtype=float)
        if matrices.shape != (3, 3, 3) or not np.isfinite(matrices).all() or any(abs(np.linalg.det(h)) < 1e-10 for h in matrices):
            raise ValueError('Need three finite nonsingular author matrices in lane order1,2,3.')
        self.matrices = matrices.copy()
        self.lanes, self.measures = {}, {}
        for source, target in [(lanes, self.lanes), (measures, self.measures)]:
            if set(source) != {1, 2, 3}:
                raise ValueError('Need exactly three lane/measurement regions with IDs1,2,3.')
            for identity, points in source.items():
                points = np.asarray(points, dtype=np.float32)
                if points.shape != (4, 2) or not np.isfinite(points).all() or not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1:
                    raise ValueError('Author region must be a finite convex quadrilateral in perimeter order.')
                target[identity] = points.copy()

    @classmethod
    def from_files(cls, matrix_path, lanes_path):
        # This parser accepts calibration RegionOfInterest only, never vehicle
        # speed/plate/frame XML. Predictor callers must not receive truth paths.
        values = np.loadtxt(matrix_path)
        if values.shape != (9, 3):
            raise ValueError('Author matrix file must contain three3x3 matrices.')
        root = ET.fromstring(Path(lanes_path).read_bytes())
        if root.tag != 'RegionOfInterest' or root.find('lanes') is None:
            raise ValueError('Expected author calibration ROI XML, not vehicle truth.')
        lanes, measures = {}, {}
        order = ['pTopLeft', 'pTopRight', 'pBotRight', 'pBotLeft']
        for tag, target in [('lane', lanes), ('measure', measures)]:
            for node in root.find('lanes').findall(tag):
                identity = int(node.attrib['id'])
                if identity in target:
                    raise ValueError('Duplicate author region ID.')
                target[identity] = [[float(node.find(key).attrib[axis]) for axis in ['x', 'y']] for key in order]
        return cls(values.reshape(3, 3, 3), lanes, measures)

    def project_road_contact(self, point, *, feature_kind, analysis_size=FULL_SIZE):
        if feature_kind != 'independently_observed_road_contact':
            return {'state': 'withheld', 'reason': 'No validated road-plane anchor; bbox bottoms, plates and body features are not interchangeable.'}
        point = np.asarray(point, dtype=float)
        size = np.asarray(analysis_size, dtype=float)
        if point.shape != (2,) or size.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(size).all() or np.any(size < 2):
            raise ValueError('Need a finite image point and analysis dimensions.')
        # Pixel-centre mapping reverses aspect-preserving analysis resize.
        full = (point+.5)*(np.asarray(FULL_SIZE)/size)-.5
        if np.any(full < 0) or np.any(full > np.asarray(FULL_SIZE)-1):
            return {'state': 'withheld', 'reason': 'point outside original1920x1080 frame'}
        matches = [identity for identity in (1, 2, 3)
                   if cv2.pointPolygonTest(self.lanes[identity], tuple(full), False) >= 0
                   and cv2.pointPolygonTest(self.measures[identity], tuple(full), False) >= 0]
        if len(matches) != 1:
            return {'state': 'withheld', 'reason': 'outside or ambiguous author measurement ROI; no extrapolation'}
        identity = matches[0]
        value = self.matrices[identity-1]@np.r_[full, 1.]
        if not np.isfinite(value).all() or value[2] <= 1e-9:
            return {'state': 'withheld', 'reason': 'invalid author homography denominator'}
        metres = value[:2]/value[2]/RECTIFIED_PIXELS_PER_METRE
        if not all(math.isfinite(v) for v in metres):
            return {'state': 'withheld', 'reason': 'invalid projected coordinate'}
        return {'state': 'projected_reference_point', 'lane_from_geometry': identity,
                'source_point_px': full.tolist(), 'rectified_point_m': metres.tolist(),
                'calibration_basis': 'author lane-specific full-resolution planar H /60pixels_per_metre',
                'lens_model': 'none separately supplied', 'plate_height_factor_applied': False}
