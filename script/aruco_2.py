import numpy as np
from .base_overlay import load_sequence, draw_plane

MARKER_ID = 2

ANIM = {
    "folder": "script/image/anim_2",
    "fps": 60,
    "speed": 1.0,
    "pause_after": 10.0,
    "pause_duration": 1.0
}

def render(frame, rvec, tvec, K, dist):
    media_frame, alpha = load_sequence(ANIM["folder"], MARKER_ID, ANIM)

    obj_pts = np.float32([
        [-0.05, 0.10, 0.02],
        [ 0.05, 0.10, 0.02],
        [ 0.05, 0.02, 0.02],
        [-0.05, 0.02, 0.02],
    ])

    return draw_plane(frame, media_frame, alpha, obj_pts, rvec, tvec, K, dist)
