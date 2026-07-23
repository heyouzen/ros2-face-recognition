from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOFaceDetector:
    """基于 YOLOv8-Face 原生track接口 人脸检测+跟踪模块。"""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.50,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到模型文件：{self.model_path}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device

        print(f"[YOLOFaceDetector] 正在加载模型：{self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("[YOLOFaceDetector] 模型加载完成")

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        纯检测，不跟踪（保留原有接口兼容旧代码）
        返回格式：
        [{"bbox": [x1, y1, x2, y2], "confidence": 0.95}]
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        result = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[dict[str, Any]] = []

        if result.boxes is None:
            return detections

        height, width = image.shape[:2]

        for box in result.boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = coordinates

            # 防止检测框越出图像边界
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(box.conf[0].cpu())

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                }
            )

        return detections

    def track(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        检测+跟踪，使用YOLO原生track，返回带track_id的人脸目标
        返回格式：
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95,
                "track_id": 1
            }
        ]
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 原生track推理，persist保留跨帧跟踪信息
        result = self.model.track(
            source=image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            verbose=False,
        )[0]

        track_targets: list[dict[str, Any]] = []
        height, width = image.shape[:2]

        if result.boxes is None or result.boxes.id is None:
            return track_targets

        for box in result.boxes:
            coordinates = box.xyxy[0].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = coordinates
            # 边界截断
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            conf = float(box.conf[0].cpu())
            tid = int(box.id[0].cpu())

            track_targets.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "track_id": tid
            })
        return track_targets

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """绘制检测框，支持显示track_id"""
        output = image.copy()

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            # 判断是否存在跟踪ID
            if "track_id" in detection:
                label = f"ID:{detection['track_id']}  {confidence:.3f}"
            else:
                label = f"Face {index}  {confidence:.3f}"

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                output,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return output
