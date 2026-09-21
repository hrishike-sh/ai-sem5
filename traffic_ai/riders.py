from collections import deque
from math import hypot

from .geometry import containment, intersection, area
from .rules import RiderEvidence


def associate_riders(motorcycles, people, heads):
    assignments = {index: [] for index in range(len(motorcycles))}
    ambiguous = set()
    for person in people:
        px1, py1, px2, py2 = person.box
        pw, ph = px2 - px1, py2 - py1
        candidates = []
        for index, motorcycle in enumerate(motorcycles):
            mx1, my1, mx2, my2 = motorcycle.box
            mw, mh = mx2 - mx1, my2 - my1
            centre = (px1 + px2) / 2
            lower_body = (px1 + .2 * pw, py1 + .5 * ph, px2 - .2 * pw, py2)
            seat_region = (mx1 - .15 * mw, my1 - .65 * mh, mx2 + .15 * mw, my2)
            body_reaches_motorcycle = py2 >= my1 + .35 * mh
            compatible_scale = ph >= .6 * mh
            if (mx1 - .1 * mw <= centre <= mx2 + .1 * mw and containment(lower_body, seat_region) >= .6
                    and py1 < my1 and body_reaches_motorcycle and compatible_scale):
                candidates.append(index)
        if len(candidates) == 1:
            if any(other is not person and intersection(person.box, other.box) / max(1, min(area(person.box), area(other.box))) > .8 for other in people):
                ambiguous.update(candidates)
            else:
                assignments[candidates[0]].append(person)
        elif candidates:
            ambiguous.update(candidates)
    result = {}
    for index, motorcycle in enumerate(motorcycles):
        associated = assignments[index]
        bare, helmet = [], []
        for person in associated:
            x1, y1, x2, y2 = person.box
            head_area = (x1, y1, x2, y1 + .4 * (y2 - y1))
            matching = [h for h in heads if h.confidence >= .7 and containment(h.box, head_area) >= .8]
            kinds = {h.label for h in matching}
            if len(matching) != 1 or not kinds <= {"helmet", "no_helmet", "no-helmet"}:
                continue
            if sum(containment(matching[0].box, (p.box[0], p.box[1], p.box[2], p.box[1] + .4 * (p.box[3] - p.box[1]))) >= .8 for p in people) != 1:
                continue
            if matching[0].label == "helmet":
                helmet.extend(matching)
            else:
                bare.extend(matching)
        result[index] = RiderEvidence(associated, bare, helmet,
                                     association_confident=bool(associated) and index not in ambiguous,
                                     visibility_complete=False)
    return result


class RiderAssociator:
    def __init__(self):
        self.pairs = {}

    def reset(self):
        self.pairs.clear()

    def observe(self, motorcycles, people, heads, timestamp, stable):
        candidates = associate_riders(motorcycles, people, heads)
        if not stable:
            self.reset()
        self.pairs = {key: samples for key, samples in self.pairs.items() if timestamp - samples[-1][0] <= .75}
        confirmed = []
        for index, motorcycle in enumerate(motorcycles):
            if not stable or motorcycle.track_id is None or not candidates[index].association_confident:
                continue
            for person in candidates[index].associated_people:
                if person.track_id is None:
                    continue
                if any(other is not person and intersection(person.box, other.box) / max(1, min(area(person.box), area(other.box))) > .8 for other in people):
                    continue
                key = (motorcycle.track_id, person.track_id)
                samples = self.pairs.setdefault(key, deque(maxlen=180))
                bike_point = motorcycle.ground_point
                person_point = person.ground_point
                scale = max(1, motorcycle.box[2] - motorcycle.box[0])
                if samples and (timestamp <= samples[-1][0] or hypot(bike_point[0] - samples[-1][1][0], bike_point[1] - samples[-1][1][1]) > 2 * scale):
                    samples.clear()
                samples.append((timestamp, bike_point, person_point))
                while samples and timestamp - samples[0][0] > 1.5:
                    samples.popleft()
                if len(samples) < 4 or timestamp - samples[0][0] < .5:
                    continue
                first = samples[0]
                bike_motion = (bike_point[0] - first[1][0], bike_point[1] - first[1][1])
                if hypot(*bike_motion) < max(5, .1 * scale):
                    continue
                residuals = [hypot((b[0] - first[1][0]) - (p[0] - first[2][0]),
                                   (b[1] - first[1][1]) - (p[1] - first[2][1])) for _, b, p in samples]
                if max(residuals) <= max(3, .08 * scale):
                    confirmed.append(person)
        exclusive_heads = [head for head in heads if sum(containment(head.box, (p.box[0], p.box[1], p.box[2], p.box[1] + .4 * (p.box[3] - p.box[1]))) >= .8 for p in people) == 1]
        result = associate_riders(motorcycles, confirmed, exclusive_heads)
        for index, evidence in result.items():
            evidence.association_confident = evidence.association_confident and candidates[index].association_confident
            evidence.association_basis = "Unique geometric match with at least 0.5 seconds of shared motion under a stable camera" if evidence.association_confident else "Shared motorcycle/person motion has not been established"
        return result
