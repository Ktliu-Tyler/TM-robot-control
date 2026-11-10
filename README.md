# HW3 ROS 2 workspace

## Standalone homework scripts (submission/)

Submission copies of the Part B-D scripts; they only need opencv-python and numpy
(submission/requirements.txt). Part A is HW_A/camera_calibration.py (see A / C / D
below); submit it as hw3_a.py. TA is the folder unpacked from the TA's hw3.zip
(with b/, c/, d/ inside):

    python3 submission/hw3_b.py TA/b/cubes.png                 # Part B: centroid + principal angle
    python3 submission/hw3_c.py TA/b/cubes.png TA/c/depth.png  # Part C: uses Part B's detector
    python3 submission/hw3_c.py TA/c/depth.png                 # Part C: depth-only fallback
    python3 submission/hw3_d.py TA/d/cubes.png TA/d/depth.png  # Part D: 6D pose (bonus)

hw3_c.py imports hw3_b.py from the same folder, so Part B and Part C share object IDs
(--depth-only keeps the original depth segmentation). hw3_d.py fits a cube model to the
color outlines and top faces on the depth table plane; it places the principal point at
the image center because Part C's (286.54, 151.36) lies outside the 240 x 240 Part D
image (--partc-K uses Part C's matrix unchanged). The package copies in
src/hw3_robot/hw3_robot are identical; keep them in sync, e.g.
cmp submission/hw3_d.py src/hw3_robot/hw3_robot/hw3_d.py.

Build on the lab ROS 2 computer with its TM driver and tm_msgs overlay sourced.
Rename team_ws to team<number>_ws for submission. From the workspace root:

    rosdep install --from-paths src --ignore-src -r -y
    colcon build --symlink-install
    source install/setup.bash

## A / C / D

Run `python camera_calibration.py` from [HW_A](HW_A/) with NumPy and OpenCV installed to reproduce the calibration. The original images, corrected examples and [camera parameters](HW_A/part_a_camera.npz) are included.
For that same camera, resolution and landscape orientation:

    ros2 run hw3_robot rectify_image --ros-args -p calibration:=/absolute/path/part_a_camera.npz

image_raw -> image_rect preserves the calibration pixel grid (no ROI crop).
depth_centers subscribes to aligned_depth (16UC1, mm). depth_poses subscribes to a
synchronized color image and aligned depth (parameters color_topic, default color_image,
and depth_topic, default aligned_depth); intrinsics come from parameters fx, fy, cx, cy
(cx/cy below 0 mean the image center). Start either with ros2 run hw3_robot followed by
its name. Results on depth_centers/result or depth_poses/result carry the input stamp and
optical frame; depth_poses angles use R = Rx(roll) Ry(pitch) Rz(yaw).
The packaged hw3_b.py, hw3_c.py and hw3_d.py are copies of submission/.

## E / fixed-view tabletop calibration

Start the TM Listen project, ROS driver, image publisher and feedback stream. The detector
is Part B's (hw3_b.detect_objects: automatic triangle threshold, noise removal, edge-aligned
angle for square blocks) and assumes separate brighter blocks on a dark background. Inspect detect_blocks/result on
lab images before motion. Each detection holds a pixel centroid and a second point along
a block edge. The stacker transforms both points to base XY to determine jaw yaw.

Create a JSON file with the following fields using measurements for the same arm, active TCP,
camera job, resolution and fixed viewing pose. pixels and base_xy_mm are matching lists
of at least four [x,y] pairs distributed around the pickup region, measured on a plane
at block-top height. Check additional points against measured base XY. If lens distortion
is significant, calibrate the TM camera and use its rectified stream for both mapping and detection.
The planar mapping applies to this single capture pose and equal-height upright blocks.

| Field | Value |
|---|---|
| camera_pose | [X,Y,Z,A,B,C] in base mm and degrees; supplied right-arm example [230,230,730,-180,0,135], left arm [350,350,730,-180,0,135] |
| image_size | [width,height] of TM image |
| pixels, base_xy_mm | matching top-plane point arrays |
| grasp_z_mm | measured TCP height for grasping a block on the table |
| clear_z_mm | transit TCP height above the completed stack and other obstacles |
| block_height_mm | measured block height |
| stack_xy_mm | reachable empty stacking site [X,Y] |
| jaw_offset_deg, stack_yaw_deg | measured jaw-to-edge yaw correction and release yaw |
| gripper_settle_s | measured open/close time plus settling margin |

    ros2 run hw3_robot detect_blocks
    ros2 run hw3_robot stack_blocks --ros-args -p calibration:=/absolute/path/lab.json

For this preview trigger a new job1 image from the fixed pose. Check all printed waypoints.
On hardware, start at a verified clear approach and run the same stack_blocks command with
-p execute:=true appended. It moves to camera_pose, requests a stamped image with
Vision_DoJob(job1), then grasps and stacks three blocks. Feedback must publish actual
tool_pose in metres and radians with the same base and TCP; verify on the lab driver.
Controller pose convention used here is Rz(C) Ry(B) Rx(A). Tool IO is module 1/type 1/pin 0.
The program checks pose arrival and service responses; grip success requires observation.
After interruption inspect arm and stack before restarting with three blocks on the table.


[Part E 中文上機手冊與架構](PART_E.md)
