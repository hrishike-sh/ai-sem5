"""Two-sided photometric support for elongated appearance proposals.

A bright band needs observed darker flanks, not merely a brightness gradient.
This is not semantic recognition: vehicle roofs and other objects can pass.
"""
import cv2
import numpy as np


def bracketed_components(rgb, proposed):
    value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(proposed)
    accepted = np.zeros_like(proposed)
    rows = []
    for identity in range(1, count):
        x, y, width, height, area = map(int, stats[identity])
        yy, xx = np.nonzero(labels[y:y+height, x:x+width] == identity)
        points = np.column_stack((xx+x, yy+y)).astype(float)
        corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
        edges = np.roll(corners, -1, axis=0)-corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        along = edge / max(np.linalg.norm(edge), 1e-9)
        normal = np.array([-along[1], along[0]])
        centre = points.mean(axis=0)
        longitudinal = (points-centre)@along
        transverse = (points-centre)@normal
        length = np.ptp(longitudinal)
        samples = []
        for position in np.linspace(np.quantile(longitudinal, .15), np.quantile(longitudinal, .85), 5):
            section = np.abs(longitudinal-position) <= max(1.5, length*.025)
            if np.count_nonzero(section) < 3:
                continue
            low, high = float(np.min(transverse[section])), float(np.max(transverse[section]))
            thickness = high-low+1
            margin = max(5., thickness*.4)
            inside = float(np.median(value[points[section, 1].astype(int), points[section, 0].astype(int)]))
            sides = []
            observed_counts = []
            observed_coordinates = []
            for offsets in (np.linspace(low-margin, low-1.5, 5), np.linspace(high+1.5, high+margin, 5)):
                coordinates = centre + position*along + offsets[:, None]*normal
                coordinates = np.round(coordinates).astype(int)
                coordinates = np.unique(coordinates, axis=0)
                valid = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < value.shape[1])
                         & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < value.shape[0]))
                coordinates = coordinates[valid]
                coordinates = coordinates[labels[coordinates[:, 1], coordinates[:, 0]] != identity]
                observed_counts.append(len(coordinates))
                observed_coordinates.append(coordinates.tolist())
                if len(coordinates) < 3:
                    sides.append(None)
                else:
                    sides.append(float(np.median(value[coordinates[:, 1], coordinates[:, 0]])))
            differences = [None if side is None else inside-side for side in sides]
            samples.append({'inside_value': inside, 'flank_values': sides, 'contrasts': differences,
                            'flank_observed_unique_pixels': observed_counts,
                            'flank_observed_xy': observed_coordinates,
                            'bracketed': all(delta is not None and delta >= 12 for delta in differences)})
        passed = len(samples) >= 3 and sum(s['bracketed'] for s in samples) >= .6*len(samples)
        if passed:
            accepted[y+yy, x+xx] = 255
        rows.append({'id': identity, 'bbox': [x, y, width, height], 'area': area,
                     'aspect': float(length/max(1, np.ptp(transverse))), 'samples': samples,
                     'retained': passed})
    return accepted, rows
