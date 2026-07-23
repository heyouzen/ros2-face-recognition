import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2
import time


class TestCameraPublisher(Node):

    def __init__(self):
        super().__init__('test_camera_pub')

        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        self.bridge = CvBridge()

        self.image_path = "/home/chen/yolo/geshirong.png"

        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise RuntimeError("图片读取失败")

        self.timer = self.create_timer(
            1.0,
            self.publish_image
        )

        self.get_logger().info(
            "虚拟摄像头启动"
        )


    def publish_image(self):

        msg = self.bridge.cv2_to_imgmsg(
            self.image,
            encoding="bgr8"
        )

        self.publisher.publish(msg)

        self.get_logger().info(
            "发布图片"
        )


def main():

    rclpy.init()

    node = TestCameraPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
