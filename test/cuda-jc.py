import cv2
print("OpenCV版本：", cv2.__version__)
print("CUDA是否可用：", cv2.cuda.getCudaEnabledDeviceCount() > 0)
print("CUDA设备数：", cv2.cuda.getCudaEnabledDeviceCount())