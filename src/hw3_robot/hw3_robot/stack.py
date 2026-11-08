import json
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tm_msgs.msg import FeedbackState
from tm_msgs.srv import SendScript, SetIO


def plan(cfg, frame):
    uv = np.asarray(cfg["pixels"], np.float32)
    xy = np.asarray(cfg["base_xy_mm"], np.float32)
    if uv.shape != xy.shape or len(uv) < 4 or uv.shape[1:] != (2,):
        raise ValueError("Supply at least four corresponding top-plane points")
    H, _ = cv2.findHomography(uv, xy)
    if H is None or not np.isfinite(H).all():
        raise ValueError("Degenerate calibration")
    hull = cv2.convexHull(uv)
    if [frame["width"], frame["height"]] != cfg["image_size"]:
        raise ValueError("Image size differs from calibration")
    if len(frame["objects"]) != 3:
        raise ValueError("Expected exactly three blocks in the capture")
    stack = np.array(cfg["stack_xy_mm"])
    grasp, clear, height = (cfg[k] for k in ("grasp_z_mm","clear_z_mm","block_height_mm"))
    if height <= 0 or clear <= grasp+3*height:
        raise ValueError("Invalid clearance")
    moves = [("io",0)]
    for level, obj in enumerate(frame["objects"]):
        pixels = np.array(obj, np.float32)
        if any(cv2.pointPolygonTest(hull, tuple(map(float,p)), False)<0 for p in pixels):
            raise ValueError("Detection outside calibrated area")
        center, end = cv2.perspectiveTransform(pixels[None], H)[0]
        if np.linalg.norm(center-stack)<2*height:
            raise ValueError("Stack site overlaps pickup area")
        theta = np.degrees(np.arctan2(*(end-center)[::-1]))
        yaw = (theta + cfg["jaw_offset_deg"] + 90)%180-90
        def pose(at,z,c):
            return ("pose",[*map(float,at),z,-180.,0.,c])
        moves.extend([pose(center,clear,yaw),pose(center,grasp,yaw),("io",1),
                      pose(center,clear,yaw),pose(stack,clear,cfg["stack_yaw_deg"]),
                      pose(stack,grasp+level*height,cfg["stack_yaw_deg"]),("io",0),
                      pose(stack,clear,cfg["stack_yaw_deg"])])
    return moves


def main():
    rclpy.init()
    node = Node("stack_blocks")
    execute = node.declare_parameter("execute",False).value
    path = node.declare_parameter("calibration","").value
    latest = {}
    def feedback(msg):
        latest["state"] = msg
        latest["received"] = time.monotonic()
    node.create_subscription(FeedbackState,"feedback_states",feedback,10)
    node.create_subscription(String,"detect_blocks/result",
                             lambda msg: latest.update(frame=json.loads(msg.data)),10)
    script = node.create_client(SendScript,"send_script")
    io = node.create_client(SetIO,"set_io")

    def wait(check, seconds, label):
        deadline = time.monotonic()+seconds
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if check():
                return
        raise RuntimeError("Timeout: "+label)

    def arrived(target):
        s = latest.get("state")
        if s is None or time.monotonic()-latest["received"]>1:
            return False
        if s.robot_error or s.e_stop or s.project_pause:
            raise RuntimeError("TM reports fault, stop or pause")
        if len(s.tool_pose)!=6:
            return False
        xyz = np.array(s.tool_pose[:3])*1000
        # Rotation matrix comparison also handles equivalent Euler triples.
        def rotation(angles):
            a,b,c = np.radians(angles)
            return (cv2.Rodrigues(np.array([0.,0.,c]))[0] @
                    cv2.Rodrigues(np.array([0.,b,0.]))[0] @
                    cv2.Rodrigues(np.array([a,0.,0.]))[0])
        delta = rotation(target[3:]).T @ rotation(np.degrees(s.tool_pose[3:]))
        return np.linalg.norm(xyz-target[:3])<2 and np.trace(delta)>1+2*np.cos(np.radians(2))

    def call(client,req):
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError("Missing service "+client.srv_name)
        future = client.call_async(req)
        wait(future.done,8,client.srv_name)
        if not future.result().ok:
            raise RuntimeError("Rejected by "+client.srv_name)

    def send(code):
        request = SendScript.Request()
        request.id = "hw3"
        request.script = code
        call(script,request)

    try:
        with open(path,encoding="utf-8-sig") as f:
            cfg = json.load(f)
        if execute:
            def move(target):
                send('PTP("CPP",'+','.join(f"{v:.4f}" for v in target)+',10,200,0,false)')
                wait(lambda: arrived(target),30,"target TCP")
            move(cfg["camera_pose"])
            latest.pop("frame",None)
            capture_time = node.get_clock().now().nanoseconds
            send("Vision_DoJob(job1)")
        def fresh():
            if "frame" not in latest:
                return False
            if not execute:
                return True
            sec,ns = latest["frame"]["stamp"]
            return sec*10**9+ns >= capture_time
        wait(fresh,20,"new image; check camera job and timestamp")
        for kind,value in plan(cfg,latest["frame"]):
            node.get_logger().info(f"{kind} {value}")
            if execute:
                if kind=="pose":
                    move(value)
                else:
                    request = SetIO.Request()
                    request.module,request.type,request.pin,request.state = 1,1,0,float(value)
                    call(io,request)
                    time.sleep(cfg["gripper_settle_s"])
    finally:
        node.destroy_node()
        rclpy.shutdown()
