import numpy as np
from .base_overlay import load_media, draw_plane

video_caps = {}

MARKER_ID = 1
MEDIA_PATH = "script/image/1.png"

def render(frame, rvec, tvec, K, dist):
    media_frame, alpha = load_media(MEDIA_PATH, video_caps, MARKER_ID)

    ## 这里定义ArUco的位置和大小
    obj_pts = np.float32([
        [0.06,  0.05, 0],
        [0.14,  0.05, 0],
        [0.14, -0.05, 0],
        [0.06, -0.05, 0],
    ])

    return draw_plane(frame, media_frame, alpha, obj_pts, rvec, tvec, K, dist)
