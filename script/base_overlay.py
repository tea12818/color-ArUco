import cv2
import numpy as np

video_caps = {}

def load_media(path, key):
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


def draw_plane(frame_u, media_frame, alpha, obj_pts, rvec, tvec, K, dist):
    h, w = media_frame.shape[:2]

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1,2).astype(np.float32)

    src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(src_pts, img_pts)

    warped = cv2.warpPerspective(cv2.UMat(media_frame), M, (frame_u.cols, frame_u.rows))
    warped_alpha = cv2.warpPerspective(cv2.UMat(alpha), M, (frame_u.cols, frame_u.rows))

    warped_alpha = cv2.merge([warped_alpha, warped_alpha, warped_alpha])
    warped_alpha = cv2.UMat(warped_alpha.get().astype(np.float32) / 255.0)

    frame_f = cv2.UMat(frame_u.get().astype(np.float32))
    warped_f = cv2.UMat(warped.get().astype(np.float32))

    out = cv2.multiply(warped_f, warped_alpha) + cv2.multiply(frame_f, 1 - warped_alpha)
    return cv2.UMat(out.get().astype(np.uint8))
