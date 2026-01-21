import cv2
import numpy as np
import os

# ===================== 配置项 =====================
# 图片存储路径（请替换为你的实际路径）
IMAGE_DIR = "./images"
# ArUco码的物理尺寸（单位：米，根据你的打印尺寸调整）
ARUCO_MARKER_SIZE = 0.1  # 示例：10cm
# ===================== 配置项结束 =====================

# 缓存加载的图片，避免重复IO
image_cache = {}
# ID与图片路径的映射（自动读取IMAGE_DIR下的id1-id30图片）
id_to_image_path = {
    i: os.path.join(IMAGE_DIR, f"id{i}.jpg") for i in range(1, 31)
}

def init_opencl() -> bool:
    """启用OpenCL加速并检查是否生效"""
    cv2.ocl.setUseOpenCL(True)
    if cv2.ocl.useOpenCL():
        print("✅ OpenCL加速已启用")
        return True
    else:
        print("⚠️ OpenCL加速未生效（可能是OpenCV编译时未启用或硬件不支持）")
        return False

def load_image(aruco_id: int) -> np.ndarray | None:
    """加载指定ID对应的图片，使用缓存避免重复加载"""
    if aruco_id not in id_to_image_path:
        print(f"❌ 无ID为{aruco_id}的图片配置")
        return None
    
    # 检查缓存
    if aruco_id in image_cache:
        return image_cache[aruco_id]
    
    # 加载图片
    img_path = id_to_image_path[aruco_id]
    if not os.path.exists(img_path):
        print(f"❌ 图片文件不存在：{img_path}")
        return None
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 图片加载失败：{img_path}")
            return None
        image_cache[aruco_id] = img
        print(f"✅ 成功加载ID{aruco_id}的图片：{img_path}")
        return img
    except Exception as e:
        print(f"❌ 加载图片出错：{e}")
        return None

def overlay_image_on_aruco(
    frame: np.ndarray,
    aruco_corners: np.ndarray,
    overlay_img: np.ndarray
) -> np.ndarray:
    """
    将图片叠加到ArUco码的位置（透视变换）
    :param frame: 原始视频帧
    :param aruco_corners: ArUco码的四个角点 (4,2)
    :param overlay_img: 要叠加的图片
    :return: 叠加后的帧
    """
    # 转换角点格式（适配透视变换）
    pts_aruco = aruco_corners.astype(np.float32)
    # 图片的四个角点（按顺序：左上、右上、右下、左下）
    h, w = overlay_img.shape[:2]
    pts_img = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts_img, pts_aruco)
    # 透视变换（将图片适配到ArUco码的位置和角度）
    warped_img = cv2.warpPerspective(overlay_img, M, (frame.shape[1], frame.shape[0]))
    
    # 创建掩码（用于只叠加图片区域，避免覆盖背景）
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_aruco.astype(np.int32), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)
    
    # 叠加图片到帧上
    frame_bg = cv2.bitwise_and(frame, mask_inv)
    frame_fg = cv2.bitwise_and(warped_img, mask)
    result = cv2.add(frame_bg, frame_fg)
    
    return result

def process_aruco_marker(
    frame: np.ndarray,
    aruco_id: int,
    aruco_corners: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> np.ndarray:
    """
    处理单个ArUco码：加载对应图片并叠加
    :param frame: 原始帧
    :param aruco_id: ArUco码ID
    :param aruco_corners: ArUco码角点
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 相机畸变系数
    :return: 处理后的帧
    """
    # 加载对应图片
    overlay_img = load_image(aruco_id)
    if overlay_img is None:
        return frame
    
    # 叠加图片到ArUco码位置
    frame = overlay_image_on_aruco(frame, aruco_corners, overlay_img)
    
    # （可选）绘制ArUco码的ID和坐标轴（便于调试位姿）
    # 计算位姿（旋转向量rvec，平移向量tvec）
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        [aruco_corners], ARUCO_MARKER_SIZE, camera_matrix, dist_coeffs
    )
    # 绘制坐标轴（长度为0.05米）
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 0.05)
    # 绘制ID文本
    center = (int(aruco_corners[:, 0].mean()), int(aruco_corners[:, 1].mean()))
    cv2.putText(
        frame, f"ID: {aruco_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 2
    )
    
    return frame