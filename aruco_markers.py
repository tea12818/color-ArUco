import cv2
import numpy as np
import os

# ==================== 全局配置（可按需调整） ====================
ARUCO_DICT = cv2.aruco.DICT_4X4_50  # ArUco字典（与识别脚本一致）
NUM_MARKERS = 30                    # 生成的码数量（0~29）
MARKER_SIZE = 200                   # 单个码的尺寸（像素）
MARGIN = 40                        # 合并生成时码之间的间距（像素）
SAVE_DIR = "aruco_markers"          # 分开生成时的保存目录

def print_menu():
    """打印交互式菜单"""
    print("="*50)
    print("          ArUco码生成工具 - 选择生成模式")
    print("="*50)
    print("1 - 分开生成：30个码，每个码生成1张独立图片")
    print("2 - 合并生成：30个码整合到1张图片中（6行5列）")
    print("="*50)

def generate_separate_markers():
    """模式1：分开生成30个独立的ArUco码图片"""
    # 创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # 加载ArUco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # 循环生成每个码
    for marker_id in range(NUM_MARKERS):
        # 生成单个码
        marker_img = np.zeros((MARKER_SIZE, MARKER_SIZE), dtype=np.uint8)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE, marker_img, 1)
        
        # 保存图片
        save_path = os.path.join(SAVE_DIR, f"aruco_{marker_id}.png")
        cv2.imwrite(save_path, marker_img)
        
        
        if (marker_id + 1) % 1 == 0:
            print(f"已生成 {marker_id + 1}/{NUM_MARKERS} 个码")
    
    print(f"\n✅ 分开生成完成！")
    print(f"📁 所有码保存至：{os.path.abspath(SAVE_DIR)}")

def generate_merged_markers():
    """模式2：将30个码合并生成在单张图片中"""
    # 计算合并后图片的尺寸（6行5列）
    ROWS = 6
    COLS = 5
    img_width = COLS * MARKER_SIZE + (COLS + 1) * MARGIN
    img_height = ROWS * MARKER_SIZE + (ROWS + 1) * MARGIN
    
    # 创建白色背景画布
    canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 加载ArUco字典
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # 循环生成并排列码
    marker_id = 0
    for row in range(ROWS):
        if marker_id >= NUM_MARKERS:
            break
        for col in range(COLS):
            # 计算当前码的位置
            x_start = MARGIN + col * (MARKER_SIZE + MARGIN)
            y_start = MARGIN + row * (MARKER_SIZE + MARGIN)
            
            # 生成单个码（单通道转3通道）
            marker_img = np.zeros((MARKER_SIZE, MARKER_SIZE), dtype=np.uint8)
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE, marker_img, 1)
            marker_img_3ch = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            
            # 粘贴到画布
            canvas[y_start:y_start+MARKER_SIZE, x_start:x_start+MARKER_SIZE] = marker_img_3ch
            
            # 标注编号
            text_x = x_start + MARKER_SIZE // 2 - 15
            text_y = y_start + MARKER_SIZE + MARGIN // 2
            cv2.putText(canvas, f"ID: {marker_id}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            marker_id += 1
            if marker_id >= NUM_MARKERS:
                break
    
    # 保存合并后的图片
    save_path = "aruco_30.png"
    cv2.imwrite(save_path, canvas)
    
    print(f"\n合并生成完成！")
    print(f"合并图片保存至：{os.path.abspath(save_path)}")
    print(f"图片尺寸：{img_width}x{img_height} 像素")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 打印菜单
    print_menu()
    
    # 交互式选择（带输入验证）
    while True:
        try:
            choice = int(input("\n请输入选择（1/2）："))
            if choice == 1:
                generate_separate_markers()
                break
            elif choice == 2:
                generate_merged_markers()
                break
            else:
                print("输入错误！请仅输入1或2")
        except ValueError:
            print("输入错误！请输入数字1或2")
    
    print("\n任务完成！")