import numpy as np
from .base_overlay import load_media, draw_plane

MARKER_ID = 2
MEDIA_PATH = "script/image/2.mp4"

def render(frame, rvec, tvec, K, dist):
    media_frame, alpha = load_media(MEDIA_PATH, MARKER_ID)

    obj_pts = np.float32([
        [-0.05, 0.10, 0.02],
        [ 0.05, 0.10, 0.02],
        [ 0.05, 0.02, 0.02],
        [-0.05, 0.02, 0.02],
    ])

    return draw_plane(frame, media_frame, alpha, obj_pts, rvec, tvec, K, dist)
