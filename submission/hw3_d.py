"""Robotics HW3 - Part D: 6D pose (X, Y, Z, roll, pitch, yaw) of each cube.

Usage:
    python3 hw3_d.py cubes.png depth.png
    python3 hw3_d.py cubes.png depth.png --debug --no-show

Idea: the cubes rest flat on a table, so their orientation is the table's
orientation plus one in-plane rotation (yaw), and their size is known from
the depth.  Each cube therefore has only 3 unknowns left (2D position on
the table + yaw), which we find by fitting a 3D cube model to the cube's
outline AND its bright top face in the color image.

Pipeline:
    1. depth -> 3D point for every valid pixel (pinhole back-projection, K)
    2. RANSAC plane fit -> table plane (normal n, offset d); height above
       the table h = n . p + d for every pixel
    3. cube size: start from the median top-face height in the depth, then
       keep the size (+-3 mm) that best fits the color image (step 5)
    4. color image -> cube outlines (silhouettes): pixels that are colorful
       (chroma) or brighter than the table; inside each outline the
       brightest part (Otsu on lightness) is the top face
    5. per cube: search position (a, b) on the table and yaw psi so that the
       projected 3D cube best overlaps the outline and the projected top
       face best overlaps the top face (sum of two IoUs, coarse-to-fine)
    6. cube frame -> rotation matrix -> roll, pitch, yaw

Camera matrix: the focal lengths come from Part C.  Part C's principal
point (286.54, 151.36) lies outside this 240 x 240 image, so the image must
be a crop and that point cannot be its principal point.  By default the
principal point is the image center; with it the fitted cube edges line up
with the photo (top-face IoU 0.88 vs 0.83).  --partc-K uses Part C's K as is.

Frames:
    camera: origin at the optical center, X right, Y down, Z forward (mm)
    cube:   origin at the cube's center; z_c points down into the table
            (for a camera looking straight down, z_c == camera Z), x_c runs
            along a cube edge, y_c = z_c x x_c.
    angles: R = Rx(roll) * Ry(pitch) * Rz(yaw), in degrees, R's columns =
            the cube axes in the camera frame.  Roll and pitch = how the
            table is tilted relative to the camera (same for every cube);
            yaw = the cube's spin on the table.  A cube looks the same after
            a 90 deg turn, so yaw is reported in (-45, 45].
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

# Camera matrix from Part C (the assignment allows reusing it for Part D)
FX, FY, CX, CY = 613.57, 613.54, 286.54, 151.36


def default_K(width, height):
    """Part C focal lengths, principal point at the image center."""
    return np.array([[FX, 0, (width - 1) / 2.0],
                     [0, FY, (height - 1) / 2.0], [0, 0, 1.0]])


def partc_K():
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])

PLANE_TOL = 3.0      # mm, a point within this distance belongs to the table
ABOVE_TABLE = 8.0    # mm, a point higher than this belongs to a cube
MIN_AREA = 100       # px, smaller blobs are noise
CHROMA_T = 20        # color image: max(BGR) - min(BGR) above this = colorful
BRIGHT_T = 20        # ... or brighter than the table's median by this much
SUPERSAMPLE = 4      # render the cube model at 4x for sub-pixel IoU

CUBE_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),      # bottom face
              (4, 5), (5, 7), (7, 6), (6, 4),      # top face
              (0, 4), (1, 5), (2, 6), (3, 7)]      # vertical edges


# --------------------------------------------------------------------------
# 1-2. Depth -> point cloud -> table plane
# --------------------------------------------------------------------------
def back_project(depth, K):
    """Organized point cloud (H x W x 3, mm) and validity mask."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    Z = depth.astype(np.float64)
    P = np.dstack([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])
    return P, depth > 0


def fit_table_plane(points, iters=500, seed=0):
    """RANSAC + least-squares refit. Returns unit normal n (pointing up,
    i.e. towards the camera) and offset d, with n . p + d = 0 on the table."""
    # fixed seed -> same result every run; RandomState for numpy < 1.17
    # (the lab's Ubuntu 18.04 / ROS 2 Dashing ships numpy 1.13)
    if hasattr(np.random, "default_rng"):
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.RandomState(seed)
    best_n, best_d, best_count = None, None, -1
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ s[0]
        count = int((np.abs(points @ n + d) < PLANE_TOL).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count

    inliers = points[np.abs(points @ best_n + best_d) < PLANE_TOL]
    c = inliers.mean(axis=0)
    n = np.linalg.svd(inliers - c, full_matrices=False)[2][2]  # least-squares
    if n[2] > 0:          # camera looks along +Z, so "up" has negative Z
        n = -n
    d = -n @ c
    rms = float(np.sqrt(np.mean((inliers @ n + d) ** 2)))
    return n, d, c, len(inliers) / len(points), rms


# --------------------------------------------------------------------------
# 4. Color image -> cube silhouettes
# --------------------------------------------------------------------------
def silhouettes(img_bgr):
    """Full cube outlines (top + side faces) from the color image."""
    I = img_bgr.astype(np.int16)
    chroma = I.max(axis=2) - I.min(axis=2)   # colorfulness, stable in shade
    value = I.max(axis=2)                    # brightness (HSV V)
    mask = (chroma > CHROMA_T) | (value > np.median(value) + BRIGHT_T)
    mask = mask.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[-2]
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    return mask, contours


def top_faces(img_bgr, contours):
    """Top face of each cube = the brightest part of its outline.
    Otsu threshold on lightness (Lab L) inside each outline; keep the
    largest piece."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0]
    k = np.ones((3, 3), np.uint8)
    tops = []
    for c in contours:
        inside = np.zeros(L.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 255, thickness=cv2.FILLED)
        t, _ = cv2.threshold(L[inside > 0].reshape(-1, 1), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        top = ((L > t) & (inside > 0)).astype(np.uint8) * 255
        top = cv2.morphologyEx(top, cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(top)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            top = (lab == biggest).astype(np.uint8) * 255
        tops.append(top)
    return tops


# --------------------------------------------------------------------------
# 5. Cube model and fitting
# --------------------------------------------------------------------------
class TableFrame:
    """2D coordinates (a, b) on the table plane, plus the 'up' normal."""

    def __init__(self, n_up, origin):
        self.n = n_up
        self.o = origin                               # a point on the plane
        e1 = np.array([1.0, 0.0, 0.0])
        e1 -= (e1 @ n_up) * n_up                      # camera x, flattened
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(-n_up, self.e1)            # (e1, e2, -n) right-handed

    def to_3d(self, a, b, height=0.0):
        return self.o + a * self.e1 + b * self.e2 + height * self.n

    def to_2d(self, p):
        q = p - self.o
        return float(q @ self.e1), float(q @ self.e2)


def cube_axes(frame, psi_deg):
    """Cube axes in the camera frame for yaw psi on the table."""
    p = math.radians(psi_deg)
    x = math.cos(p) * frame.e1 + math.sin(p) * frame.e2
    z = -frame.n                                     # down into the table
    y = np.cross(z, x)
    return x, y, z


def cube_corners(frame, a, b, psi_deg, s):
    """8 corners (camera frame): index = 4*top + 2*(y side) + (x side)."""
    x, y, _ = cube_axes(frame, psi_deg)
    base = frame.to_3d(a, b)
    pts = []
    for top in (0, 1):
        for sy in (-1, 1):
            for sx in (-1, 1):
                pts.append(base + sx * s / 2 * x + sy * s / 2 * y
                           + top * s * frame.n)
    return np.array(pts)


def project(K, pts):
    uvw = (K @ pts.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def ray_plane(K, u, v, frame, height):
    """3D point where the camera ray through pixel (u, v) meets the plane
    lifted by `height` above the table."""
    ray = np.linalg.solve(K, np.array([u, v, 1.0]))
    p0 = frame.to_3d(0, 0, height)
    t = (frame.n @ p0) / (frame.n @ ray)
    return t * ray


# coarse-to-fine grid search: (position step mm, half-range mm,
#                               yaw step deg, yaw half-range deg)
COARSE = [(2.0, 8.0, 3.0, 45.0), (0.5, 2.0, 0.5, 3.0)]
FINE = [(0.1, 0.5, 0.1, 0.5)]
LOCAL = [(0.25, 0.5, 0.25, 0.5)]   # quick re-fit while trying cube sizes


class CubeFitter:
    """Scores a cube pose (a, b, psi, s) against one cube's outline and top
    face: IoU(projected cube, outline) + IoU(projected top face, top face)."""

    def __init__(self, K, frame, sil_mask, top_mask, contour):
        self.K, self.frame, self.contour = K, frame, contour
        x0, y0, w, h = cv2.boundingRect(contour)
        pad = 12
        self.x0, self.y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        x1 = min(self.x0 + w + 2 * pad, sil_mask.shape[1])
        y1 = min(self.y0 + h + 2 * pad, sil_mask.shape[0])

        def crop_up(mask):  # crop to the cube and supersample (0/1 uint8)
            m = (mask[self.y0:y1, self.x0:x1] > 0).astype(np.uint8)
            return np.repeat(np.repeat(m, SUPERSAMPLE, 0), SUPERSAMPLE, 1)

        self.ref_sil, self.ref_top = crop_up(sil_mask), crop_up(top_mask)
        self.area_sil = cv2.countNonZero(self.ref_sil)
        self.area_top = cv2.countNonZero(self.ref_top)
        self.canvas = np.zeros(self.ref_sil.shape, np.uint8)

    def _iou(self, poly, ref, ref_area):
        """IoU of the filled convex polygon with a reference mask."""
        self.canvas[:] = 0
        hull = cv2.convexHull(np.round(poly * 16).astype(np.int32))
        cv2.fillConvexPoly(self.canvas, hull, 1, cv2.LINE_8, shift=4)
        inter = cv2.countNonZero(cv2.bitwise_and(self.canvas, ref))
        union = cv2.countNonZero(self.canvas) + ref_area - inter
        return inter / float(union) if union else 0.0

    def score(self, a, b, psi, s):
        uv = project(self.K, cube_corners(self.frame, a, b, psi, s))
        uv = (uv - [self.x0 - 0.5, self.y0 - 0.5]) * SUPERSAMPLE
        i_sil = self._iou(uv, self.ref_sil, self.area_sil)      # whole cube
        i_top = self._iou(uv[4:], self.ref_top, self.area_top)  # corners 4-7
        return i_sil + i_top, i_sil, i_top

    def start(self, s):
        """Initial guess: ray through the outline's centroid, half a cube up."""
        M = cv2.moments(self.contour)
        p = ray_plane(self.K, M["m10"] / M["m00"], M["m01"] / M["m00"],
                      self.frame, s / 2)
        a, b = self.frame.to_2d(p - (s / 2) * self.frame.n)
        return (a, b, 0.0)

    def search(self, guess, s, stages):
        """Grid search around guess (a, b, psi). Returns ((a, b, psi), score)."""
        best_p, best = guess, self.score(*guess, s)
        for step, half, ystep, yrange in stages:
            ca, cb, cpsi = best_p
            offs = np.arange(-half, half + 1e-9, step)
            for psi in np.arange(cpsi - yrange, cpsi + yrange + 1e-9, ystep):
                for da in offs:
                    for db in offs:
                        val = self.score(ca + da, cb + db, psi, s)
                        if val[0] > best[0]:
                            best_p, best = (ca + da, cb + db, float(psi)), val
        return best_p, best


def fit_all(fitters, s0, fixed_size=False):
    """Fit every cube. Unless the size is fixed, also pick the cube size
    (s0 +- 3 mm) that best explains all outlines and top faces together."""
    coarse = [f.search(f.start(s0), s0, COARSE)[0] for f in fitters]
    s = s0
    if not fixed_size:
        best_total = -1.0
        for cand in np.arange(s0 - 3.0, s0 + 3.0 + 1e-9, 0.25):
            total = sum(f.search(g, cand, LOCAL)[1][0]
                        for f, g in zip(fitters, coarse))
            if total > best_total:
                best_total, s = total, float(cand)
        coarse = [f.search(g, s, COARSE[1:])[0]
                  for f, g in zip(fitters, coarse)]
    results = [f.search(g, s, FINE) for f, g in zip(fitters, coarse)]
    return results, s


# --------------------------------------------------------------------------
# 6. Rotation matrix -> roll, pitch, yaw
# --------------------------------------------------------------------------
def rpy_from_R(R):
    """R = Rx(roll) Ry(pitch) Rz(yaw) -> (roll, pitch, yaw) in degrees.

    Read right to left: spin the cube about its own up/down axis (yaw),
    then tilt it (pitch about y, roll about x) to match the table.  For
    cubes lying on one table, roll and pitch are the table's tilt (the same
    for every cube) and yaw is each cube's spin on the table.
    """
    pitch = math.atan2(R[0, 2], math.hypot(R[0, 0], R[0, 1]))
    roll = math.atan2(-R[1, 2], R[2, 2])
    yaw = math.atan2(-R[0, 1], R[0, 0])
    return tuple(math.degrees(t) for t in (roll, pitch, yaw))


def R_from_rpy(roll, pitch, yaw):
    r, p, y = (math.radians(t) for t in (roll, pitch, yaw))
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def pose_from_fit(frame, a, b, psi, s):
    """Cube center + rotation. Of the 4 equivalent yaws (cube symmetry),
    keep the one with the smallest |yaw|."""
    center = frame.to_3d(a, b, s / 2)
    best = None
    for k in range(4):
        x, y, z = cube_axes(frame, psi + 90 * k)
        R = np.column_stack([x, y, z])
        roll, pitch, yaw = rpy_from_R(R)
        if best is None or abs(yaw) < abs(best[1][2]):
            best = (R, (roll, pitch, yaw), psi + 90 * k)
    R, rpy, psi_k = best
    return center, R, rpy, psi_k


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------
def estimate_poses(img_bgr, depth, K, size=None):
    P, valid = back_project(depth, K)
    n, d, c, inlier_frac, rms = fit_table_plane(P[valid])
    height = np.where(valid, P @ n + d, 0.0)
    frame = TableFrame(n, c)

    # cube size: median height of top-face points of every raised blob
    raised = ((height > ABOVE_TABLE) & valid).astype(np.uint8) * 255
    raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(raised)
    top_heights = []
    for i in range(1, nlab):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        hb = height[lab == i]
        top_heights.append(np.median(hb[hb > np.percentile(hb, 90) - 5]))
    s_depth = float(np.median(top_heights)) if top_heights else float("nan")
    n_blobs = len(top_heights)

    sil, contours = silhouettes(img_bgr)
    tops = top_faces(img_bgr, contours)
    fitters = [CubeFitter(K, frame, sil, top, cnt)
               for cnt, top in zip(contours, tops)]
    s0 = size if size else (s_depth if np.isfinite(s_depth) else 25.0)
    results, s = fit_all(fitters, s0, fixed_size=bool(size))

    cubes = []
    for f, ((a, b, psi), (_, iou, iou_top)) in zip(fitters, results):
        center, R, rpy, psi_k = pose_from_fit(frame, a, b, psi % 90.0, s)
        M = cv2.moments(f.contour)
        cubes.append(dict(center=center, R=R, rpy=rpy, iou=iou,
                          iou_top=iou_top,
                          corners=cube_corners(frame, a, b, psi_k, s),
                          uv=(M["m10"] / M["m00"], M["m01"] / M["m00"])))

    cubes.sort(key=lambda o: (-round(o["uv"][1]), o["uv"][0]))
    for i, o in enumerate(cubes):
        o["id"] = i
    info = dict(normal=n, inlier_frac=inlier_frac, plane_rms=rms,
                tilt=math.degrees(math.acos(abs(n[2]))), size=s,
                size_depth=s_depth, n_blobs=n_blobs)
    top_all = np.max(np.stack(tops), axis=0) if tops else np.zeros_like(sil)
    return cubes, info, sil, height, top_all


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_result(img_bgr, cubes, K, scale=3):
    h0, w0 = img_bgr.shape[:2]
    canvas = cv2.resize(img_bgr, (w0 * scale, h0 * scale),
                        interpolation=cv2.INTER_CUBIC)

    def px(p3):
        uv = project(K, np.atleast_2d(p3))[0]
        return (int(round((uv[0] + 0.5) * scale - 0.5)),
                int(round((uv[1] + 0.5) * scale - 0.5)))

    for o in cubes:
        pts = [px(p) for p in o["corners"]]
        for i, j in CUBE_EDGES:
            cv2.line(canvas, pts[i], pts[j], (0, 255, 255), 1, cv2.LINE_AA)
        c = o["center"]
        L = 0.9 * np.linalg.norm(o["corners"][1] - o["corners"][0])
        for k, col in enumerate([(0, 0, 255), (0, 200, 0), (255, 80, 0)]):
            cv2.arrowedLine(canvas, px(c), px(c + L * o["R"][:, k]), col, 2,
                            cv2.LINE_AA, tipLength=0.2)
        cv2.circle(canvas, px(c), 3, (255, 255, 255), -1, cv2.LINE_AA)
        u, v = px(o["corners"][7])
        cv2.putText(canvas, f"ID {o['id']}", (u + 6, v - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

    W = canvas.shape[1]
    line_h, fs = 22, 0.5
    lines = [f"ID {o['id']}: XYZ ({o['center'][0]:.1f}, {o['center'][1]:.1f},"
             f" {o['center'][2]:.1f}) mm   RPY ({o['rpy'][0]:.1f},"
             f" {o['rpy'][1]:.1f}, {o['rpy'][2]:.1f}) deg" for o in cubes]
    lines.append("axes: x red, y green, z blue (into table);  "
                 "R = Rx(roll) Ry(pitch) Rz(yaw)")
    header = np.full((line_h * len(lines) + 14, W, 3), 255, np.uint8)
    for i, txt in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        color = (0, 0, 0) if i < len(cubes) else (90, 90, 90)
        cv2.putText(header, txt, ((W - tw) // 2, 20 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    return np.vstack([header, canvas])


def draw_debug(img_bgr, sil, top, height, valid, cubes, K, scale=2):
    """Three panels: outlines + top faces | height above table | fit."""
    h0, w0 = img_bgr.shape[:2]
    up = lambda im: cv2.resize(im, (w0 * scale, h0 * scale),
                               interpolation=cv2.INTER_NEAREST)
    p1 = cv2.cvtColor(sil, cv2.COLOR_GRAY2BGR) // 2   # outline = gray
    p1[top > 0] = (255, 255, 255)                     # top face = white
    p1 = up(p1)
    hm = np.clip(height / 40.0 * 255, 0, 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm[~valid] = 0                          # black = no depth measured
    p2 = up(hm)
    p3 = up(img_bgr)
    for o in cubes:
        uv = project(K, o["corners"]) * scale
        for i, j in CUBE_EDGES:
            cv2.line(p3, tuple(int(t) for t in uv[i]),
                     tuple(int(t) for t in uv[j]), (0, 255, 255), 1,
                     cv2.LINE_AA)
    for p, t in [(p1, "outlines (gray), top faces (white)"), (p2, "height above table, 0-40 mm (black: no depth)"),
                 (p3, "fitted cube models")]:
        cv2.putText(p, t, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([p1, p2, p3])


# --------------------------------------------------------------------------
def hw3_d(img_path, depth_path, K, size=None, out=None, show=True,
          debug=False):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # keep 16-bit mm
    if img is None or depth is None:
        print("Error: cannot read the color or depth image")
        return None
    if depth.shape[:2] != img.shape[:2]:
        print("Error: color and depth images must have the same size")
        return None

    if K is None:
        K = default_K(img.shape[1], img.shape[0])
    cubes, info, sil, height, top = estimate_poses(img, depth, K, size)

    print(f"K: fx {K[0, 0]:.2f}, fy {K[1, 1]:.2f}, "
          f"cx {K[0, 2]:.2f}, cy {K[1, 2]:.2f}")
    print(f"table plane: {info['inlier_frac'] * 100:.0f}% of depth pixels, "
          f"fit RMS {info['plane_rms']:.2f} mm, tilted "
          f"{info['tilt']:.1f} deg from the optical axis")
    print(f"cube size: {info['size']:.2f} mm (best image fit; depth alone: "
          f"{info['size_depth']:.1f} mm from {info['n_blobs']} raised blobs)")
    print(f"{'ID':>2} {'X':>8} {'Y':>8} {'Z':>8}   {'roll':>7} {'pitch':>7}"
          f" {'yaw':>7}   IoU outline/top")
    for o in cubes:
        X, Y, Z = o["center"]
        r, p, y = o["rpy"]
        print(f"{o['id']:>2} {X:8.1f} {Y:8.1f} {Z:8.1f}   {r:7.1f} {p:7.1f}"
              f" {y:7.1f}   {o['iou']:.3f} / {o['iou_top']:.3f}")
    print("(mm, camera frame; degrees, R = Rx(roll) Ry(pitch) Rz(yaw))")

    stem = os.path.splitext(os.path.basename(img_path))[0]
    result = draw_result(img, cubes, K)
    out = out or f"{stem}_d_result.png"
    cv2.imwrite(out, result)
    print(f"saved: {out}")
    if debug:
        dbg = f"{stem}_d_debug.png"
        cv2.imwrite(dbg, draw_debug(img, sil, top, height, depth > 0, cubes, K))
        print(f"saved: {dbg}")

    if show and sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("no display found: skipped the window (result PNG is saved)")
        show = False
    if show:
        try:
            win = "Part D result (q / ESC to quit)"
            cv2.imshow(win, result)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        except (cv2.error, KeyboardInterrupt):
            pass
    return cubes


def main():
    ap = argparse.ArgumentParser(description="HW3 Part D: 6D pose of cubes")
    ap.add_argument("image", help="color image, e.g. cubes.png")
    ap.add_argument("depth", help="aligned 16-bit depth image (mm)")
    ap.add_argument("--K", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"),
                    default=None, help="camera intrinsics (default: Part C "
                    "focal lengths, principal point at the image center)")
    ap.add_argument("--partc-K", action="store_true",
                    help="use Part C's camera matrix exactly as given")
    ap.add_argument("--size", type=float, default=None,
                    help="fix the cube edge in mm (default: best fit to the "
                    "image, starting from the depth)")
    ap.add_argument("--out", default=None, help="output image path")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--debug", action="store_true",
                    help="also save a 3-panel image of the intermediate steps")
    args = ap.parse_args()

    if args.K:
        fx, fy, cx, cy = args.K
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    else:
        K = partc_K() if args.partc_K else None   # None -> image center
    res = hw3_d(args.image, args.depth, K, args.size, args.out,
                show=not args.no_show, debug=args.debug)
    sys.exit(0 if res is not None else 1)


if __name__ == "__main__":
    main()
