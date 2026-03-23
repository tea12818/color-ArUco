import cv2
import cv2.aruco as aruco
import numpy as np
import importlib

# ========= 开启 OpenCL =========
cv2.ocl.setUseOpenCL(True)
print("OpenCL available:", cv2.ocl.haveOpenCL())
print("OpenCL enabled :", cv2.ocl.useOpenCL())

# ========= 相机参数（临时用）=========
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)
marker_length = 0.05

# ========= ArUco =========
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict)

# ========= 加载脚本 =========
script_modules = {}
for i in range(1, 31):
    try:
        script_modules[i] = importlib.import_module(f"script.aruco_{i}")
        print(f"[INFO] Loaded script for ID {i}")
    except ModuleNotFoundError:
        pass

cap = cv2.VideoCapture(0)
show_debug = True

print("按 ESC 退出，按 \\ 开关调试显示")

while True:
    ret, frame_cpu = cap.read()
    if not ret:
        break

    frame = cv2.UMat(frame_cpu)  # 送入 GPU

    # ===== ArUco 检测（需CPU）=====
    corners, ids, _ = detector.detectMarkers(frame_cpu)

    if ids is not None:
        # 直接使用 solvePnP 兼容所有版本
        obj_pts = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0],
        ], dtype=np.float32)

        rvecs = []
        tvecs = []
        for c in corners:
            img_pts = c.reshape(-1, 1, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                rvec = np.zeros((3,1), dtype=np.float32)
                tvec = np.zeros((3,1), dtype=np.float32)
            rvecs.append(rvec)
            tvecs.append(tvec)

        rvecs = np.array(rvecs, dtype=np.float32)
        tvecs = np.array(tvecs, dtype=np.float32)

        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = rvecs[i], tvecs[i]

            if show_debug:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i in range(len(ids)):
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length*0.5)


            if marker_id in script_modules:
                frame = script_modules[marker_id].render(
                    frame, rvec, tvec, camera_matrix, dist_coeffs
                )

    cv2.imshow("AR System (OpenCL)", frame.get())

    key = cv2.waitKey(1) & 0xFF

    # 按 \ 键切换调试显示
    if key == ord('\\'):
        show_debug = not show_debug


cap.release()
cv2.destroyAllWindows()
