import json
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from .hw3_c import find_centers_from_depth
from .hw3_d import estimate


def run(name, topic, encoding, process):
    rclpy.init()
    node = Node(name)
    bridge = CvBridge()
    pub = node.create_publisher(String, name + "/result", 10)
    def receive(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
            result = process(img)
            payload = dict(frame=msg.header.frame_id, width=msg.width, height=msg.height,
                           stamp=[msg.header.stamp.sec, msg.header.stamp.nanosec], objects=result)
            pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        except Exception as exc:
            node.get_logger().error(str(exc))

    node.create_subscription(Image, topic, receive, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def centers():
    run("depth_centers", "aligned_depth", "16UC1", lambda d: find_centers_from_depth(d)[0])


def poses():
    def detect(depth):
        return [dict(xyz_mm=p.tolist(), rpy_deg=r.tolist()) for p,r in estimate(depth,depth)]
    run("depth_poses", "aligned_depth", "16UC1", detect)


def rectify():
    rclpy.init()
    node = Node("rectify_image")
    bridge = CvBridge()
    with np.load(node.declare_parameter("calibration","").value) as data:
        k, d, size = data["K"], data["D"], tuple(data["size"])
    pub = node.create_publisher(Image,"image_rect",10)
    def receive(msg):
        if (msg.width,msg.height) != size:
            node.get_logger().error("Input resolution or orientation differs from Part A")
            return
        img = bridge.imgmsg_to_cv2(msg,"bgr8")
        output = bridge.cv2_to_imgmsg(cv2.undistort(img,k,d),"bgr8")
        output.header = msg.header
        pub.publish(output)
    node.create_subscription(Image,"image_raw",receive,qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()



