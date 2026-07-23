import argparse
import json
from pathlib import Path

import cv2

from detector_yolo import YOLOFaceDetector


PROJECT_ROOT = Path.home() / "yolo"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8s-face-lindevs.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 YOLOv8-Face 检测图片中的人脸"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入图片路径",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detected.jpg",
        help="画框结果保存路径",
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "detections.json",
        help="检测结果 JSON 保存路径",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLOv8-Face 模型路径",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="置信度阈值",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备，例如 cpu、0",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入图片不存在：{args.input}")

    image = cv2.imread(str(args.input))

    if image is None:
        raise RuntimeError(
            f"OpenCV 无法读取图片：{args.input}\n"
            "请检查文件格式或文件是否损坏。"
        )

    detector = YOLOFaceDetector(
        model_path=str(args.model),
        conf_threshold=args.conf,
        device=args.device,
    )

    detections = detector.detect(image)

    print(f"\n检测到 {len(detections)} 张人脸")

    for index, detection in enumerate(detections, start=1):
        print(
            f"人脸 {index}: "
            f"bbox={detection['bbox']}, "
            f"confidence={detection['confidence']:.4f}"
        )

    annotated_image = detector.draw_detections(image, detections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.output), annotated_image):
        raise RuntimeError(f"结果图片保存失败：{args.output}")

    result_data = {
        "input_image": str(args.input),
        "face_count": len(detections),
        "detections": detections,
    }

    args.json_output.write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n画框图片已保存：{args.output}")
    print(f"JSON结果已保存：{args.json_output}")


if __name__ == "__main__":
    main()
def get_face_feature(img_path):
     detector = YOLOFaceDetector(DEFAULT_MODEL)
     feat = get_face_feature(str(img_path))
     return feat
