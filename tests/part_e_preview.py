"""Offline ROS 2 Humble integration test. Run in a dedicated ROS domain."""
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1" or not os.environ.get("ROS_DOMAIN_ID"):
        raise SystemExit("Set ROS_LOCALHOST_ONLY=1 and an isolated ROS_DOMAIN_ID")
    rclpy.init()
    node = rclpy.create_node("hw3_fixture")
    pub = node.create_publisher(Image, "techman_image", 10)
    img = np.zeros((200, 200, 3), np.uint8)
    for x in (20, 65, 110):
        cv2.rectangle(img, (x, 50), (x+15, 65), (255,255,255), -1)
    msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
    mock = "--mock-tm" in __import__("sys").argv
    scripts, outputs = [], []
    if mock:
        from tm_msgs.msg import FeedbackState
        from tm_msgs.srv import SendScript, SetIO
        state = FeedbackState()
        state.tool_pose = [0.0]*6
        feedback = node.create_publisher(FeedbackState, "feedback_states", 10)
        def send(req, res):
            scripts.append(req.script)
            if req.script.startswith("PTP"):
                values = np.array(req.script.split(",")[1:7], dtype=float)
                state.tool_pose = list(values[:3]/1000) + list(np.radians(values[3:]))
            res.ok = True
            return res
        def grip(req, res):
            outputs.append(req.state)
            res.ok = (req.module,req.type,req.pin) == (1,1,0)
            return res
        node.create_service(SendScript,"send_script",send)
        node.create_service(SetIO,"set_io",grip)
    pts = [[0,0],[199,0],[199,199],[0,199]]
    cfg = dict(camera_pose=[0]*6, pixels=pts, base_xy_mm=pts, image_size=[200,200],
               stack_xy_mm=[160,160], grasp_z_mm=10, clear_z_mm=100, block_height_mm=15,
               jaw_offset_deg=0, stack_yaw_deg=0, gripper_settle_s=0.01)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/"fixture.json"
        path.write_text(json.dumps(cfg))
        prefix = ["ros2","run","hw3_robot"]
        detector = subprocess.Popen(prefix+["detect_blocks"], start_new_session=True)
        try:
            with subprocess.Popen(prefix+["stack_blocks","--ros-args","-p",f"calibration:={path}", "-p", "execute:="+str(mock).lower()],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as controller:
                deadline = time.monotonic()+30
                while controller.poll() is None and time.monotonic()<deadline:
                    msg.header.stamp = node.get_clock().now().to_msg()
                    if mock:
                        feedback.publish(state)
                        rclpy.spin_once(node,timeout_sec=0.01)
                    if not mock or "Vision_DoJob(job1)" in scripts:
                        pub.publish(msg)
                    time.sleep(.15)
                log = controller.communicate(timeout=5)[0]
                assert controller.returncode == 0, log
                assert log.count("pose [") == 18 and log.count("io ") == 7, log
                for z in (10,25,40):
                    assert f"160.0, 160.0, {z}," in log, log
                if mock:
                    assert len(scripts)==20 and scripts[1]=="Vision_DoJob(job1)", scripts
                    assert outputs == [0,1,0,1,0,1,0], outputs
                print("PASS: ROS image -> three detections -> 18 poses and 7 IO actions")
        finally:
            import signal
            os.killpg(detector.pid, signal.SIGTERM)
            detector.wait()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

