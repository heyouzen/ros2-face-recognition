from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path.home() / "yolo"
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"


print("[FaceFeature] 正在加载 InsightFace 模型...")

face_analyzer = FaceAnalysis(
    name="buffalo_l",
    root=str(INSIGHTFACE_ROOT),
    allowed_modules=[
        "detection",
        "recognition",
    ],
    providers=["CPUExecutionProvider"],
)

face_analyzer.prepare(
    ctx_id=-1,
    det_size=(320, 320),
)

print("[FaceFeature] InsightFace加载完成")


def get_face_feature(image_or_path):
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
        source_name = "视频画面"
    else:
        img = cv2.imread(str(image_or_path))
        source_name = str(image_or_path)

    if img is None or img.size == 0:
        raise RuntimeError(f"无法读取图像: {source_name}")

    faces = face_analyzer.get(img)

    if not faces:
        raise RuntimeError(f"没有检测到人脸: {source_name}")

    face = max(
        faces,
        key=lambda f: float(f.bbox[2] - f.bbox[0])
        * float(f.bbox[3] - f.bbox[1]),
    )

    return np.asarray(
        face.normed_embedding,
        dtype=np.float32,
    )
