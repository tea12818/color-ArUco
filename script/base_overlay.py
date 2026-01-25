import cv2
import numpy as np
import glob
import time
import os
video_caps = {}
image_sequences = {}
sequence_states = {}



def load_sequence(folder, key, playback):
    if key not in image_sequences:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        image_sequences[key] = files

    if key not in sequence_states:
        sequence_states[key] = {"start_time": time.time()}

    files = image_sequences[key]
    state = sequence_states[key]

    fps = playback.get("fps", 30)
    speed = playback.get("speed", 1.0)
    pause_after = playback.get("pause_after", None)
    pause_duration = playback.get("pause_duration", 0)

    elapsed = (time.time() - state["start_time"]) * speed
    frame_index = int(elapsed * fps)

    if pause_after and elapsed >= pause_after:
        time.sleep(pause_duration)
        state["start_time"] = time.time()
        frame_index = 0

    frame_index = frame_index % len(files)

    img = cv2.imread(files[frame_index], cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 4:
        return img[:,:,:3], img[:,:,3]
    else:
        alpha = np.ones(img.shape[:2], dtype=np.uint8)*255
        return img, alpha


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
    if img is None:
        raise FileNotFoundError(f"Media not found: {path}")

    if img.shape[2] == 4:
        return img[:,:,:3], img[:,:,3]
    else:
        alpha = np.ones(img.shape[:2], dtype=np.uint8)*255
        return img, alpha


def draw_plane(frame_u, media_frame, alpha, obj_pts, rvec, tvec, K, dist):
    # ⚠️ UMat 尺寸稳定获取方式
    frame_cpu = frame_u.get()
    H, W = frame_cpu.shape[:2]

    h, w = media_frame.shape[:2]

    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1,2).astype(np.float32)

    src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(src_pts, img_pts)

    warped = cv2.warpPerspective(cv2.UMat(media_frame), M, (W, H))
    warped_alpha = cv2.warpPerspective(cv2.UMat(alpha), M, (W, H))

    # 转 float
    warped_f = cv2.UMat(warped.get().astype(np.float32))
    frame_f  = cv2.UMat(frame_cpu.astype(np.float32))

    alpha_cpu = warped_alpha.get().astype(np.float32) / 255.0
    alpha_3 = cv2.merge([alpha_cpu, alpha_cpu, alpha_cpu])
    alpha_f = cv2.UMat(alpha_3)

    one = cv2.UMat(np.ones_like(alpha_3, dtype=np.float32))

    part1 = cv2.multiply(warped_f, alpha_f)
    part2 = cv2.multiply(frame_f, cv2.subtract(one, alpha_f))

    out = cv2.add(part1, part2)

    return cv2.UMat(out.get().astype(np.uint8))

