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
from .hw3_b import detect_objects
from .hw3_c import find_centers_from_depth
from . import hw3_d


def run(name, topic, encoding, process):
    rclpy.init()
    node = Node(name)
    bridge = CvBridge()
    pub = node.create_publisher(String, name + "/result", 10)
    options = {}
    if name == "detect_blocks":
        # 0 = automatic threshold (Part B); e.g. 50 = fixed gray threshold
        options["threshold"] = node.declare_parameter("threshold", 0).value
        options["min_area"] = node.declare_parameter("min_area", 90).value
        options["max_area"] = node.declare_parameter("max_area", 1000000).value

    def receive(msg):
        try:
            img = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
            result = process(img, **options)
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
    """Part D on live data: synchronized color + aligned depth -> cube poses.
    Intrinsics: parameters fx, fy, cx, cy (cx/cy < 0 -> image center)."""
    import message_filters
    rclpy.init()
    node = Node("depth_poses")
    bridge = CvBridge()
    p = lambda name, value: node.declare_parameter(name, value).value
    color_topic, depth_topic = p("color_topic", "color_image"), p("depth_topic", "aligned_depth")
    fx, fy, cx, cy = p("fx", hw3_d.FX), p("fy", hw3_d.FY), p("cx", -1.0), p("cy", -1.0)
    pub = node.create_publisher(String, "depth_poses/result", 10)

    def receive(color_msg, depth_msg):
        try:
            img = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            h, w = depth.shape[:2]
            K = np.array([[fx, 0, cx if cx >= 0 else (w - 1) / 2.0],
                          [0, fy, cy if cy >= 0 else (h - 1) / 2.0], [0, 0, 1.0]])
            cubes = hw3_d.estimate_poses(img, depth, K)[0]
            objects = [dict(xyz_mm=[float(t) for t in o["center"]],
                            rpy_deg=[float(t) for t in o["rpy"]]) for o in cubes]
            payload = dict(frame=depth_msg.header.frame_id, width=w, height=h,
                           stamp=[depth_msg.header.stamp.sec, depth_msg.header.stamp.nanosec],
                           rpy_convention="R = Rx(roll) Ry(pitch) Rz(yaw)", objects=objects)
            pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        except Exception as exc:
            node.get_logger().error(str(exc))

    subs = [message_filters.Subscriber(node, Image, t, qos_profile=qos_profile_sensor_data)
            for t in (color_topic, depth_topic)]
    sync = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.05)
    sync.registerCallback(receive)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def detect_blocks(img, threshold=0, min_area=90, max_area=1000000):
    """Part B detector (hw3_b.detect_objects: noise removal, moment centroid,
    edge-aligned angle for square blocks). threshold <= 0 picks the gray
    threshold automatically (triangle method); > 0 uses that fixed value.
    Each block -> [centroid, point half a block edge away along that edge]."""
    fixed = int(threshold) if threshold > 0 else None
    objects = []
    for obj in detect_objects(img, thresh=fixed, min_area=min_area)[0]:
        if obj["area"] > max_area:
            continue
        center = np.array(obj["centroid"], float)
        half = 0.5 * np.sqrt(obj["area"])            # half the block edge, px
        a = np.radians(obj["angle"])
        end = center + half * np.array([np.cos(a), np.sin(a)])
        objects.append([center.tolist(), end.tolist()])
    return objects


def blocks():
    run("detect_blocks","techman_image","bgr8",detect_blocks)

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


