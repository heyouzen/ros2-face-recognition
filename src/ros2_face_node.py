import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge

import cv2
import json
import tempfile
import os

from src.face_engine import FaceEngine


bridge = CvBridge()


class FaceROSNode(Node):

    def __init__(self):

        super().__init__("face_rec_node")


        # 初始化识别引擎
        self.engine = FaceEngine()


        # 订阅摄像头
        self.sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.img_cb,
            10
        )


        # 发布识别结果
        self.pub = self.create_publisher(
            String,
            "/face/recognition_result",
            10
        )


        self.get_logger().info(
            "【接收节点】启动成功"
        )



    def img_cb(self, msg):

        try:

            # ROS Image -> OpenCV
            frame = bridge.imgmsg_to_cv2(
                msg,
                "bgr8"
            )


            h,w,_ = frame.shape


            # 临时保存图片给face_engine
            tmp_path = "/tmp/current_face.jpg"

            cv2.imwrite(
                tmp_path,
                frame
            )


            # 人脸识别
            results = self.engine.recognize_image(
                tmp_path
            )


            # ======================
            #       可视化
            # ======================

            for face in results:


                x1,y1,x2,y2 = face["bbox"]


                name = face["name"]

                score = face["similarity"]


                status = face["status"]



                # 画框

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )


                # 显示文字

                text = (
                    f"{name} "
                    f"{score:.2f}"
                )


                cv2.putText(
                    frame,
                    text,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )



            # 显示窗口

            cv2.imshow(
                "Face Recognition",
                frame
            )


            cv2.waitKey(1)



            # ======================
            # ROS结果发布
            # ======================


            send_data = {

                "width":w,

                "height":h,

                "detect_objs":results

            }


            self.pub.publish(

                String(
                    data=json.dumps(
                        send_data,
                        ensure_ascii=False
                    )
                )

            )


            self.get_logger().info(
                f"识别结果:{results}"
            )


        except Exception as e:


            self.get_logger().error(
                f"处理异常:{e}"
            )



def main():

    rclpy.init()


    node = FaceROSNode()


    try:

        rclpy.spin(node)


    except KeyboardInterrupt:

        pass


    finally:

        node.destroy_node()

        rclpy.shutdown()


        cv2.destroyAllWindows()



if __name__ == "__main__":

    main()
