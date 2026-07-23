import cv2
from pathlib import Path

from src.detector_yolo import YOLOFaceDetector
from src.recognize_face import recognize


class FaceEngine:

    def __init__(self):

        # YOLO 人脸检测
        self.detector = YOLOFaceDetector(
            model_path="/home/chen/yolo/models/yolov8s-face-lindevs.pt",
            device="cpu"
        )

        print("[FaceEngine] 初始化完成")



    # 原始图片接口
    # 用于单张图片测试
    def recognize_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("图片读取失败")


        return self._recognize(image)

    # ROS接口
    # 输入opencv frame
    def recognize_frame(self, frame):

        return self._recognize(frame)


    # 公共识别函数
    def _recognize(self, image):

        results = []


        # 1. YOLO检测人脸
        faces = self.detector.detect(image)


        # 没有人脸
        if len(faces) == 0:
            return results



        # 2. 对每张脸进行ArcFace识别

        for idx, face in enumerate(faces):

            x1, y1, x2, y2 = face["bbox"]


            # 防止越界
            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)


            if x2 <= x1 or y2 <= y1:
                continue



            # 裁剪人脸
            crop = image[y1:y2, x1:x2]


            # 临时保存
            crop_path = (
                Path("/home/chen/yolo/outputs")
                / f"ros_face_{idx}.jpg"
            )


            cv2.imwrite(
                str(crop_path),
                crop
            )


            # ArcFace识别
            rec = recognize(
                str(crop_path)
            )


            results.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "det_confidence":
                        float(face["confidence"]),

                    "name":
                        rec["name"],

                    "similarity":
                        float(rec["similarity"]),

                    "status":
                        rec["status"]
                }
            )


        return results



if __name__ == "__main__":


    engine = FaceEngine()


    result = engine.recognize_image(
        "/home/chen/yolo/geshirong.png"
    )


    print(result)
