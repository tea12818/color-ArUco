import cv2
import cv2.aruco as aruco
import numpy as np
import importlib

# ================= 相机参数（临时通用参数，用于先跑通） =================
# 分辨率假设 640x480，如果你摄像头不是这个分辨率问题也不大
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

marker_length = 0.05  # ArUco 实际边长：5cm（按你的实际改）

# ================= ArUco 检测器 =================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ================= 动态加载 script 文件夹里的 ArUco 脚本 =================
script_modules = {}
for i in range(1, 31):
    try:
        module = importlib.import_module(f"script.aruco_{i}")
        script_modules[i] = module
        print(f"[INFO] Loaded script for ArUco ID {i}")
    except ModuleNotFoundError:
        pass

# ================= 打开摄像头 =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ AR 系统启动，按 ESC 退出")

# ================= 主循环 =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # 估计每个 marker 的姿态
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in script_modules:
                module = script_modules[marker_id]

                # 调用对应脚本的 render 函数
                frame = module.render(
                    frame,
                    rvecs[i],
                    tvecs[i],
                    camera_matrix,
                    dist_coeffs
                )

    cv2.imshow("AR System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
