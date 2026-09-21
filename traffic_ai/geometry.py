import math


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def containment(inner, outer):
    return intersection(inner, outer) / max(area(inner), 1e-9)


def point_in_polygon(point, polygon):
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = a
        bx, by = b
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-8 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def signed_distance(point, line):
    (ax, ay), (bx, by) = line
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6:
        raise ValueError("A boundary must have distinct endpoints")
    return ((bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)) / length


def crossed_segment(previous, current, line, margin=2.0):
    a, b = signed_distance(previous, line), signed_distance(current, line)
    if not (a < -margin and b > margin):
        return False
    t = -a / (b - a)
    point = (previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1]))
    p, q = line
    dot = (point[0] - p[0]) * (q[0] - p[0]) + (point[1] - p[1]) * (q[1] - p[1])
    return 0 <= dot <= (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
