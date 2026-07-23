import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from detector_yolo import YOLOFaceDetector


PROJECT_ROOT = Path.home() / "yolo"

IMAGE_PATH = PROJECT_ROOT / "test_data" / "geshirong.jpg"
YOLO_MODEL_PATH = (
    PROJECT_ROOT / "models" / "yolov8s-face-lindevs.pt"
)

CROP_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "geshirong_face_crop.jpg"
)

EMBEDDING_OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "geshirong_embedding.npy"
)

METADATA_OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "geshirong_metadata.json"
)


def expand_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    ratio: float = 0.15,
) -> list[int]:
    """将人脸框向四周扩展，同时保证不越界。"""

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    pad_x = int(width * ratio)
    pad_y = int(height * ratio)

    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    ]


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"找不到测试照片：{IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            f"OpenCV无法读取照片：{IMAGE_PATH}"
        )

    # 1. 使用YOLOv8-Face寻找人脸
    detector = YOLOFaceDetector(
        model_path=str(YOLO_MODEL_PATH),
        conf_threshold=0.35,
        iou_threshold=0.50,
        image_size=640,
        device="cpu",
    )

    detections = detector.detect(image)

    if not detections:
        raise RuntimeError("YOLOv8-Face没有检测到人脸")

    # 当前照片只有一个人，选择置信度最高的人脸
    best_detection = max(
        detections,
        key=lambda item: item["confidence"],
    )

    original_bbox = best_detection["bbox"]
    detector_confidence = best_detection["confidence"]

    image_height, image_width = image.shape[:2]

    expanded_bbox = expand_bbox(
        original_bbox,
        image_width,
        image_height,
        ratio=0.15,
    )

    x1, y1, x2, y2 = expanded_bbox
    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise RuntimeError("人脸裁剪结果为空")

    CROP_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(CROP_OUTPUT_PATH),
        face_crop,
    ):
        raise RuntimeError("人脸裁剪图保存失败")

    print(f"YOLO检测框：{original_bbox}")
    print(f"扩展后人脸框：{expanded_bbox}")
    print(f"检测置信度：{detector_confidence:.4f}")
    print(f"人脸裁剪图：{CROP_OUTPUT_PATH}")

    # 2. 加载InsightFace/ArcFace
    print("\n正在加载InsightFace模型……")
    print("首次运行可能需要下载模型，请耐心等待。")

    insightface_root = (
        PROJECT_ROOT / "models" / "insightface"
    )

    face_analyzer = FaceAnalysis(
        name="buffalo_l",
        root=str(insightface_root),
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

    # 3. 在YOLO裁出的人脸区域中获取关键点及特征
    faces = face_analyzer.get(face_crop)

    if not faces:
        raise RuntimeError(
            "InsightFace无法在YOLO裁剪区域中定位人脸"
        )

    # 如果检测到多个，选择面积最大的一个
    selected_face = max(
        faces,
        key=lambda face: (
            float(face.bbox[2] - face.bbox[0])
            * float(face.bbox[3] - face.bbox[1])
        ),
    )

    embedding = np.asarray(
        selected_face.normed_embedding,
        dtype=np.float32,
    )

    if embedding.ndim != 1:
        embedding = embedding.reshape(-1)

    EMBEDDING_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDING_OUTPUT_PATH,
        embedding,
    )

    metadata = {
        "person_id": "geshirong",
        "name": "葛世荣",
        "role": "中国工程院院士",
        "source_image": str(IMAGE_PATH),
        "face_crop": str(CROP_OUTPUT_PATH),
        "embedding_file": str(EMBEDDING_OUTPUT_PATH),
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_norm": float(
            np.linalg.norm(embedding)
        ),
        "detector_confidence": float(
            detector_confidence
        ),
        "bbox": original_bbox,
    }

    METADATA_OUTPUT_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nArcFace特征提取成功")
    print(f"特征维度：{embedding.shape}")
    print(
        "特征L2范数："
        f"{np.linalg.norm(embedding):.6f}"
    )
    print(f"特征文件：{EMBEDDING_OUTPUT_PATH}")
    print(f"元数据文件：{METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
