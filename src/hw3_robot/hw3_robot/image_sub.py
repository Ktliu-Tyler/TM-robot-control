import json
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
def detect_blocks(img, threshold=50, min_area=90, max_area=1000000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    objects = []
    for contour in contours:
        if not min_area <= cv2.contourArea(contour) <= max_area:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        edges = np.roll(box,-1,axis=0)-box
        edge = edges[np.argmax(np.linalg.norm(edges,axis=1))]
        moments = cv2.moments(contour)
        center = np.array([moments["m10"],moments["m01"]])/moments["m00"]
        objects.append([center.tolist(), (center+edge/2).tolist()])
    return objects



def main():
    rclpy.init()
    node = Node("image_sub")
    bridge = CvBridge()
    pub = node.create_publisher(String, "detect_blocks/result", 10)
    options = {key: node.declare_parameter(key, default).value
               for key, default in [("threshold", 50), ("min_area", 90), ("max_area", 1000000)]}

    def receive(msg):
        try:
            objects = detect_blocks(bridge.imgmsg_to_cv2(msg, "bgr8"), **options)
            payload = dict(frame=msg.header.frame_id, width=msg.width, height=msg.height,
                           stamp=[msg.header.stamp.sec, msg.header.stamp.nanosec], objects=objects)
            pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        except Exception as exc:
            node.get_logger().error(str(exc))

    node.create_subscription(Image, "techman_image", receive, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except Exception:
        if rclpy.ok():
            raise
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

