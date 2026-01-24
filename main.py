import cv2
import cv2.aruco as aruco
import numpy as np
import importlib

# ================= 相机参数 =================
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_length = 0.05  # 5cm

# ================= ArUco 检测器 =================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ================= 加载脚本模块 =================
script_modules = {}
for i in range(1, 31):
    try:
        module = importlib.import_module(f"script.aruco_{i}")
        script_modules[i] = module
        print(f"[INFO] Loaded script for ArUco ID {i}")
    except ModuleNotFoundError:
        pass

# ================= 摄像头 =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ AR 系统启动，按 ESC 退出，按 \\ 键开关调试显示")

# 调试显示开关
show_debug = True

# ================= 主循环 =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i]

            # ===== 调试信息：方框 + 坐标轴 =====
            if show_debug:
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # ===== 调用对应 ArUco 脚本 =====
            if marker_id in script_modules:
                module = script_modules[marker_id]
                frame = module.render(frame, rvec, tvec, camera_matrix, dist_coeffs)

    cv2.imshow("AR System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('\\'):  # 反斜杠键切换调试显示
        show_debug = not show_debug
        print(f"[DEBUG] 调试显示：{'开启' if show_debug else '关闭'}")

cap.release()
cv2.destroyAllWindows()
