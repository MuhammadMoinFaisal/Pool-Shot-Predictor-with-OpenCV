import math
import numpy as np
def findintersection(movement_dir, top_left, bot_right, target):
    theta = math.atan2(movement_dir[0], movement_dir[1])
    a = math.sin(theta)
    b = math.cos(theta)
    w = bot_right[0]
    h = bot_right[1]
    rx = top_left[0]
    ry = top_left[1]

    t1 = (w - target[0]) / a
    t2 = (h - target[1]) / b
    t3 = (rx - target[0]) / a
    t4 = (ry - target[1]) / b

    curr_t = 9999999
    for t in [t1, t2, t3, t4]:
        if t > 0 and t < curr_t:
            curr_t = t

    if curr_t == t1 or curr_t == t3:
        normal = np.array([1, 0])
    else:
        normal = np.array([0, 1])
    intersection = (int(target[0] + curr_t * a), int(target[1] + curr_t * b))
    return intersection, normal


def insidepocket(pockets, pred):
    for p in pockets:
        if np.linalg.norm(p - pred) < 40:
            return 1
    return 0
