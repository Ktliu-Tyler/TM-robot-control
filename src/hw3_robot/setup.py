from setuptools import setup
setup(name="hw3_robot", version="0.1.0", packages=["hw3_robot"],
      data_files=[("share/ament_index/resource_index/packages", ["resource/hw3_robot"]),
                  ("share/hw3_robot", ["package.xml"])],
      entry_points={"console_scripts": [
          "image_sub = hw3_robot.image_sub:main",
          "send_script = hw3_robot.send_script:main",
          "rectify_image = hw3_robot.nodes:rectify",
          "depth_centers = hw3_robot.nodes:centers",
          "depth_poses = hw3_robot.nodes:poses",
          "detect_blocks = hw3_robot.image_sub:main",
          "stack_blocks = hw3_robot.send_script:main"]})

