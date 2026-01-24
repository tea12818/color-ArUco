# media_overlay.py
import cv2
import numpy as np

class MediaOverlay:
    def __init__(self, media_map, marker_length=0.05, side="right"):
        """
        media_map: dict {id: "path/to/image_or_video"}
        marker_length: ArUco 实际边长（单位：米）
        side: "right" 或 "left"
        """
        self.media_map = media_map
        self.marker_length = marker_length
        self.side = side
        self.video_caps = {}

    def _get_media_frame(self, marker_id):
        path = self.media_map.get(marker_id)
        if path is None:
            return None

        # 视频
        if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            if marker_id not in self.video_caps:
                self.video_caps[marker_id] = cv2.VideoCapture(path)

            cap = self.video_caps[marker_id]
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            return frame

        # 图片
        img = cv2.imread(path)
        return img

    def draw_overlay(self, frame, marker_id, rvec, tvec, camera_matrix, dist_coeffs):
        media_frame = self._get_media_frame(marker_id)
        if media_frame is None:
            return frame

        h, w, _ = media_frame.shape

        # 定义“媒体平面”在 marker 坐标系下的四个角点
        offset = self.marker_length * 1.2  # 离二维码一点距离
        size = self.marker_length * 1.5    # 屏幕大小

        if self.side == "right":
            obj_pts = np.float32([
                [ self.marker_length/2 + offset,  size/2, 0],
                [ self.marker_length/2 + offset + size,  size/2, 0],
                [ self.marker_length/2 + offset + size, -size/2, 0],
                [ self.marker_length/2 + offset, -size/2, 0],
            ])
        else:  # left
            obj_pts = np.float32([
                [-self.marker_length/2 - offset - size,  size/2, 0],
                [-self.marker_length/2 - offset,         size/2, 0],
                [-self.marker_length/2 - offset,        -size/2, 0],
                [-self.marker_length/2 - offset - size, -size/2, 0],
            ])

        # 投影到图像坐标
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        img_pts = img_pts.reshape(-1, 2)

        # 媒体图像四个角
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # 透视变换
        M = cv2.getPerspectiveTransform(src_pts, img_pts.astype(np.float32))
        warped = cv2.warpPerspective(media_frame, M, (frame.shape[1], frame.shape[0]))

        # 创建 mask
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, img_pts.astype(int), 255)

        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(warped, warped, mask=mask)

        combined = cv2.add(bg, fg)
        return combined
