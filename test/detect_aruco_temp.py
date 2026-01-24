import cv2
import numpy as np

# ==================== 1. 检查CUDA支持 ====================
def check_cuda_support():
    """检查OpenCV的CUDA支持情况"""
    has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"OpenCV CUDA 支持: {has_cuda}")
    print(f"CUDA设备数量: {cv2.cuda.getCudaEnabledDeviceCount()}")
    if has_cuda:
        print(f"当前GPU: {cv2.cuda.getDevice()}")
    return has_cuda

# 初始化CUDA支持标志
CUDA_AVAILABLE = check_cuda_support()

# ==================== 2. 核心配置 ====================
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_LENGTH = 0.1  # 实际打印的ArUco码边长（米）
rotation_angle = 0   # 初始旋转角度（0°=无旋转）

# 相机内参（基础值，旋转后动态适配）
camera_matrix = np.array([[900, 0, 640],
                          [0, 900, 480],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 初始化ArUco检测器
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ==================== 3. 图像旋转函数（适配不同角度） ====================
def rotate_frame(frame, angle, use_cuda=False):
    """
    按指定角度旋转图像
    :param frame: 输入帧（CPU/GPU Mat）
    :param angle: 旋转角度（0/90/180/270）
    :param use_cuda: 是否使用GPU加速
    :return: 旋转后的帧（CPU Mat）
    """
    if angle == 0:
        return frame.download() if use_cuda else frame
    
    # GPU旋转逻辑
    if use_cuda:
        gpu_frame = frame
        if angle == 90:  # 向左/逆时针90°
            rotated = gpu_frame.transpose(cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:  # 旋转180°
            rotated = cv2.cuda.flip(gpu_frame, -1)  # 水平+垂直翻转
        elif angle == 270:  # 向右/顺时针90°
            rotated = gpu_frame.transpose(cv2.ROTATE_90_CLOCKWISE)
        return rotated.download()
    # CPU旋转逻辑
    else:
        if angle == 90:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return rotated

# ==================== 4. 摄像头初始化 ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ 无法打开摄像头")

# 设置摄像头参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# 打印操作提示
print("\n✅ 可旋转版ArUco识别已启动")
print("📌 操作说明：")
print("   - 按 '-' 键：画面向左旋转90°")
print("   - 按 '+' 键：画面向右旋转90°")
print("   - 按 'f' 键：查看当前帧率")
print("   - 按 'q' 键：退出程序")

# ==================== 5. 帧率计算 ====================
fps_counter = 0
fps_start = cv2.getTickCount()
fps_display = 0

# ==================== 6. 实时识别循环 ====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    # ---------- 图像旋转处理 ----------
    if CUDA_AVAILABLE:
        # CPU帧上传到GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        # GPU旋转 + 下载到CPU
        frame_rot = rotate_frame(gpu_frame, rotation_angle, use_cuda=True)
        # GPU转灰度图
        gpu_frame_rot = cv2.cuda_GpuMat()
        gpu_frame_rot.upload(frame_rot)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame_rot, cv2.COLOR_BGR2GRAY)
        gray = gpu_gray.download()
    else:
        # CPU旋转 + 转灰度
        frame_rot = rotate_frame(frame, rotation_angle, use_cuda=False)
        gray = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)

    # ---------- ArUco码检测 ----------
    corners, ids, rejected = detector.detectMarkers(gray)

    # ---------- 标注编号和XYZ坐标 ----------
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame_rot, corners, ids)
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            corner = corners[i]
            
            # 估计位姿（XYZ坐标）
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner, MARKER_LENGTH, camera_matrix, dist_coeffs
            )
            
            # 绘制XYZ坐标轴
            cv2.drawFrameAxes(frame_rot, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH/2)
            
            # 计算中心坐标
            corner_points = corner[0]
            center_x = int((corner_points[0][0] + corner_points[2][0]) / 2)
            center_y = int((corner_points[0][1] + corner_points[2][1]) / 2)
            
            # 显示编号和XYZ
            x, y, z = round(tvec[0][0][0], 3), round(tvec[0][0][1], 3), round(tvec[0][0][2], 3)
            cv2.putText(frame_rot, f"ID: {marker_id}", (center_x-20, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_rot, f"X:{x}m Y:{y}m Z:{z}m", (center_x-60, center_y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ---------- 帧率 & 旋转角度显示 ----------
    # 帧率计算
    fps_counter += 1
    if fps_counter >= 10:
        fps_end = cv2.getTickCount()
        fps_display = (fps_counter * cv2.getTickFrequency()) / (fps_end - fps_start)
        fps_counter = 0
        fps_start = cv2.getTickCount()
    
    # 显示帧率、GPU状态、旋转角度
    cv2.putText(frame_rot, f"FPS: {fps_display:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame_rot, f"GPU: {CUDA_AVAILABLE}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame_rot, f"Rotate: {rotation_angle}°", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # ---------- 显示与键盘控制 ----------
    cv2.imshow("ArUco Detection (Rotatable)", frame_rot)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        print(f"当前帧率: {fps_display:.1f} FPS | 当前旋转角度: {rotation_angle}°")
    elif key == ord('-'):  # 向左旋转90°
        rotation_angle = (rotation_angle + 90) % 360
        print(f"🔄 画面向左旋转90°，当前角度: {rotation_angle}°")
    elif key == ord('+') or key == ord('='):  # 向右旋转90°（兼容+和=键）
        rotation_angle = (rotation_angle - 90) % 360
        print(f"🔄 画面向右旋转90°，当前角度: {rotation_angle}°")

# ==================== 7. 释放资源 ====================
cap.release()
cv2.destroyAllWindows()
print(f"\n✅ 程序已退出 | 最终旋转角度: {rotation_angle}°")