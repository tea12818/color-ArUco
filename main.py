# main.py
import cv2
import cv2.aruco as aruco
import numpy as np
from media_overlay import MediaOverlay

# ========= 相机参数（⚠️强烈建议你换成自己标定得到的）=========
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 如果没标定先这样

marker_length = 0.05  # 5cm

# ========= ArUco 字典 =========
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ========= ID 对应的媒体 =========
media_map = {
    1: "media/1.jpg",
    2: "media/2.mp4",
    3: "media/3.jpg",
    # 一直加到 30
}

overlay = MediaOverlay(media_map, marker_length=marker_length, side="right")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rvecs[i], tvecs[i]

            # 画坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # 调用副程序叠加媒体
            frame = overlay.draw_overlay(
                frame, marker_id, rvec, tvec, camera_matrix, dist_coeffs
            )

    cv2.imshow("Aruco AR Player", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()
