import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import sys

class VideoPubNode(Node):
    def __init__(self, video_path):
        super().__init__("video_sim_publisher")
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.cap = cv2.VideoCapture(video_path)
        self.rate = self.create_rate(30)
        self.get_logger().info(f"视频文件打开状态: {self.cap.isOpened()}")

    def run_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().info("视频播放完毕")
                break
            h, w, c = frame.shape
            img_msg=Image()
            img_msg.header.stamp = (
                self.get_clock().now().to_msg()
            )

            img_msg.header.frame_id = "camera"
            img_msg.height = h
            img_msg.width = w
            img_msg.encoding = "bgr8"
            img_msg.step = w * c
            img_msg.data = frame.tobytes()
            self.pub.publish(img_msg)
            self.get_logger().info("发布一帧图像")
            rclpy.spin_once(
                self,
                timeout_sec=0
            )

            self.rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("用法: python3 src/video_to_ros_pub.py 视频路径")
        return
    node = VideoPubNode(sys.argv[1])
    node.run_loop()
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
