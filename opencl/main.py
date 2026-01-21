import cv2
import numpy as np
import aruco_functions as af

# ===================== 相机参数配置（关键） =====================
# 示例相机内参矩阵（请替换为你自己标定的参数，手机/摄像头需单独标定）
# 标定工具：OpenCV自带的calibrateCamera函数
CAMERA_MATRIX = np.array([
    [900, 0, 640],    # fx, 0, cx
    [0, 900, 480],    # 0, fy, cy
    [0, 0, 1]         # 0, 0, 1
], dtype=np.float32)

# 示例畸变系数（k1, k2, p1, p2, k3）
DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
# ===================== 相机参数结束 =====================

def main():
    # 1. 初始化OpenCL加速
    af.init_opencl()
    
    # 2. 加载ArUco字典（4x4_50）
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # 3. 打开视频流（0为默认摄像头，可替换为视频文件路径）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头/视频流")
        return
    
    # 设置摄像头分辨率（可选，根据硬件调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("📹 视频流已启动，按 'q' 退出")
    
    # 4. 实时检测循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧")
            break
        
        # 5. 检测ArUco码
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # 6. 处理检测到的ArUco码（仅处理ID1-30）
        if ids is not None:
            for i, aruco_id in enumerate(ids.flatten()):
                # 过滤ID范围（1-30）
                if 1 <= aruco_id <= 30:
                    # 获取当前ArUco码的角点（4个点）
                    aruco_corners = corners[i][0]
                    # 调用副程序处理图片叠加
                    frame = af.process_aruco_marker(
                        frame, aruco_id, aruco_corners,
                        CAMERA_MATRIX, DIST_COEFFS
                    )
        
        # 7. 显示结果
        cv2.imshow("ArUco Marker Detection", frame)
        
        # 8. 退出逻辑（按q键）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 9. 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()