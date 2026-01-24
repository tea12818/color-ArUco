import cv2
import numpy as np

def load_media(path, video_caps, key):
    if path.lower().endswith((".mp4",".avi",".mov",".mkv")):
        if key not in video_caps:
            video_caps[key] = cv2.VideoCapture(path)
        cap = video_caps[key]
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        alpha = np.ones(frame.shape[:2], dtype=np.uint8)*255
        return frame, alpha

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        return img[:,:,:3], img[:,:,3]
    else:
        alpha = np.ones(img.shape[:2], dtype=np.uint8)*255
        return img, alpha


def draw_plane(frame, media_frame, alpha, obj_pts, rvec, tvec, K, dist):
    h, w = media_frame.shape[:2]
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1,2).astype(np.float32)

    src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(src_pts, img_pts)

    warped = cv2.warpPerspective(media_frame, M, (frame.shape[1], frame.shape[0]))
    warped_alpha = cv2.warpPerspective(alpha, M, (frame.shape[1], frame.shape[0]))

    warped_alpha = np.dstack([warped_alpha/255.0]*3)
    return (warped*warped_alpha + frame*(1-warped_alpha)).astype(np.uint8)
