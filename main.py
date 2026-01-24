import cv2
import cv2.aruco as aruco
import importlib

# 相机参数（建议换成标定值）
camera_matrix = ...
dist_coeffs = ...

marker_length = 0.05

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict)

# 动态加载脚本
script_modules = {}
for i in range(1, 31):
    try:
        module = importlib.import_module(f"script.aruco_{i}")
        script_modules[i] = module
    except ModuleNotFoundError:
        pass

cap = cv2.VideoCapture(0)

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
            if marker_id in script_modules:
                module = script_modules[marker_id]
                frame = module.render(
                    frame,
                    rvecs[i],
                    tvecs[i],
                    camera_matrix,
                    dist_coeffs
                )

    cv2.imshow("AR System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
