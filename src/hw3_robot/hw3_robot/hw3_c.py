#!/usr/bin/env python3
"""Robotics HW3 Part C: independently estimate 3D cube centers from aligned depth.

Normal submission usage:
    python hw3_c.py cubes.png depth.png

Depth-only development/testing (when Part B's RGB file is not yet available):
    python hw3_c.py depth.png

The depth segmentation below is intentionally tuned to the supplied HW3-C scene:
table at approximately 590-595 mm and cube tops at approximately 566-568 mm.
It is NOT a general object detector; replace find_centers_from_depth() with the
Part B RGB object detector if you need to handle different scenes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np

# Provided by the instructor for the RGB-D sample images (not from Part A).
FX = 613.57
FY = 613.54
CX = 286.54
CY = 151.36


def find_centers_from_depth(depth: np.ndarray, min_area: int = 100,
                            separation_mm: float = 12.0):
    """Detect closer cube surfaces than the background, ignoring invalid depth 0."""
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        raise ValueError("No valid depth values found.")

    # For the provided scene the background occupies most pixels, so the
    # median is a good table-depth estimate.  This is a scene assumption.
    table_depth = float(np.median(valid_depth))
    threshold_mm = table_depth - separation_mm
    mask = np.uint8((depth > 0) & (depth < threshold_mm)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    objects = []
    for comp_id in range(1, count):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        u, v = (float(x) for x in centers[comp_id])
        # Median over all valid pixels of the object is more robust than
        # taking one depth sample at its centroid (which could be zero).
        object_depth = depth[labels == comp_id]
        object_depth = object_depth[object_depth > 0]
        if object_depth.size == 0:
            continue
        z = float(np.median(object_depth))  # depth.png is in millimeters
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        objects.append(dict(u=u, v=v, x=x, y=y, z=z, area=area))

    # Consistent IDs: top-to-bottom, and left-to-right at similar heights.
    objects.sort(key=lambda obj: (round(obj["v"] / 30), obj["u"]))
    return objects, mask, table_depth, threshold_mm


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    valid = depth[depth > 0]
    lo, hi = np.percentile(valid, [1, 99])
    scaled = np.clip((depth.astype(np.float32) - lo) * 255.0 / max(hi-lo, 1),
                     0, 255).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    color[depth == 0] = 0
    return color


def draw_results(background: np.ndarray, objects) -> np.ndarray:
    # 2x output improves readability while maintaining image coordinates.
    out = cv2.resize(background, None, fx=2, fy=2,
                     interpolation=cv2.INTER_NEAREST)
    height, width = out.shape[:2]
    for index, obj in enumerate(objects):
        px, py = round(obj["u"] * 2), round(obj["v"] * 2)
        cv2.circle(out, (px, py), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
        label = (f'ID{index}: ({obj["x"]:.1f}, {obj["y"]:.1f}, '
                 f'{obj["z"]:.0f}) mm')
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               0.48, 1)
        text_x = int(np.clip(px + 10, 6, max(6, width - tw - 8)))
        text_y = int(np.clip(py - 13, th + 8, height - 6))
        cv2.rectangle(out, (text_x-3, text_y-th-5),
                      (text_x+tw+3, text_y+baseline+3), (25, 25, 25), -1)
        cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Estimate cube 3D camera coordinates")
    parser.add_argument("images", nargs="+", metavar="IMAGE",
                        help="depth.png OR cubes.png depth.png")
    parser.add_argument("-o", "--output", default="part_c_result.png",
                        help="Annotated result image")
    args = parser.parse_args()
    if len(args.images) == 1:
        rgb_path = None
        depth_path = args.images[0]
    elif len(args.images) == 2:
        rgb_path, depth_path = args.images
    else:
        parser.error("Supply depth.png or cubes.png depth.png")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("Depth image must be a readable 16-bit single-channel PNG.")
    if rgb_path is None:
        background = depth_visualization(depth)
        print("[Depth-only mode] RGB cubes.png not supplied; drawing on depth visualization.")
    else:
        background = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(f"Cannot read RGB image: {rgb_path}")
        if background.shape[:2] != depth.shape:
            raise ValueError("RGB and depth must be aligned and have same dimensions.")

    objects, mask, table, threshold = find_centers_from_depth(depth)
    if not objects:
        raise RuntimeError("No objects detected. Check the depth image and threshold.")
    output = draw_results(background, objects)
    cv2.imwrite(args.output, output)
    cv2.imwrite(str(Path(args.output).with_name('part_c_mask.png')), mask)

    print(f"Depth shape: {depth.shape[1]} x {depth.shape[0]}, uint16, mm")
    print(f"Estimated table: {table:.1f} mm; object threshold: < {threshold:.1f} mm")
    print("Coordinates are in CAMERA frame (X right, Y down, Z forward)")
    print(" ID | pixel(u,v)       | X(mm)    Y(mm)    Z(mm)")
    for i, obj in enumerate(objects):
        print(f'{i:3d} | ({obj["u"]:6.1f},{obj["v"]:6.1f}) | '
              f'{obj["x"]:7.1f} {obj["y"]:8.1f} {obj["z"]:8.1f}')
    print(f"Saved: {args.output}, part_c_mask.png")


if __name__ == "__main__":
    main()
