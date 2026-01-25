import cv2
import os

video_path = "input.m4v"
output_dir = "script/image/anim_2"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_path = os.path.join(output_dir, f"{frame_id:04d}.png")
    cv2.imwrite(out_path, frame)
    frame_id += 1

print("Done! Total frames:", frame_id)
cap.release()
