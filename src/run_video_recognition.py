import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from src.detector_yolo import YOLOFaceDetector
from src.face_feature import get_face_feature
from src.recognize_face import cosine_similarity, load_face_lib


PROJECT_ROOT = Path.home() / "yolo"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8s-face-lindevs.pt"
MAX_HISTORY_LEN = 10


def expand_bbox(bbox, width, height, ratio=0.15):
    x1, y1, x2, y2 = bbox
    padding_x = int((x2 - x1) * ratio)
    padding_y = int((y2 - y1) * ratio)

    return [
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(width, x2 + padding_x),
        min(height, y2 + padding_y),
    ]


def match_face(face_image, face_lib, threshold):
    embedding = get_face_feature(face_image)
    best_name = "unknown"
    best_score = -1.0

    for name, saved_feature in face_lib.items():
        score = cosine_similarity(embedding, saved_feature)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        best_name = "unknown"

    return {
        "name": best_name,
        "similarity": best_score,
        "status": "known" if best_name != "unknown" else "stranger",
    }


def get_stable_result(history_list):
    if not history_list:
        return None
    name_list = [item["name"] for item in history_list]
    counter = Counter(name_list)
    most_name = counter.most_common(1)[0][0]
    match_items = [i for i in history_list if i["name"] == most_name]
    avg_sim = sum([i["similarity"] for i in match_items]) / len(match_items)
    status = "known" if most_name != "unknown" else "stranger"
    return {
        "name": most_name,
        "similarity": round(avg_sim, 3),
        "status": status
    }


def draw_result(frame, item):
    x1, y1, x2, y2 = item["bbox"]
    result = item["stable_result"]
    track_id = item["track_id"]

    if result["status"] == "known":
        color = (0, 255, 0)
        display_name = result["name"].split("_")[0]
    elif result["status"] == "stranger":
        color = (0, 0, 255)
        display_name = "unknown"
    else:
        color = (0, 165, 255)
        display_name = "face_unclear"

    label = f"ID:{track_id} {display_name}  {result.get('similarity', 0.0):.3f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def run_video(input_path, output_path, model_path, threshold, interval, display):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"视频不存在：{input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    face_lib = load_face_lib()
    detector = YOLOFaceDetector(model_path=str(model_path), device="cpu")
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 修复点：去除嵌套的capture.get()
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    frame_number = 0
    known_count = 0
    stranger_count = 0
    unclear_count = 0
    last_results = []
    start_time = time.time()
    track_history = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1

            if frame_number == 1 or frame_number % interval == 0:
                detections = detector.track(frame)
                current_results = []
                for detection in detections:
                    track_id = detection["track_id"]
                    original_bbox = detection["bbox"]
                    cx1, cy1, cx2, cy2 = expand_bbox(original_bbox, width, height)
                    face_crop = frame[cy1:cy2, cx1:cx2]
                    try:
                        result = match_face(face_crop, face_lib, threshold)
                        if result["status"] == "known":
                            known_count += 1
                        else:
                            stranger_count += 1
                    except RuntimeError:
                        unclear_count += 1
                        result = {"name": "face_unclear", "similarity": 0.0, "status": "unclear"}

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(result)
                    if len(track_history[track_id]) > MAX_HISTORY_LEN:
                        track_history[track_id].pop(0)
                    stable_res = get_stable_result(track_history[track_id])
                    current_results.append({"bbox": original_bbox, "track_id": track_id, "stable_result": stable_res})

                last_results = current_results

            for item in last_results:
                draw_result(frame, item)

            cv2.putText(frame, f"Frame: {frame_number}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)

            if display:
                preview = frame
                if width > 1280:
                    scale = 1280 / width
                    preview = cv2.resize(frame, None, fx=scale, fy=scale)
                cv2.imshow("Video Face Recognition - press Q to stop", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        elapsed = time.time() - start_time
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "frames": frame_number,
        "video_fps": round(fps, 2),
        "processing_fps": round(processing_fps, 2),
        "real_time": processing_fps >= fps,
        "known_results": known_count,
        "stranger_results": stranger_count,
        "unclear_results": unclear_count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    run_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        threshold=args.threshold,
        interval=max(1, args.interval),
        display=args.display,
    )


if __name__ == "__main__":
    main()
