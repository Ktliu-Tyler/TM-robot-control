"""Robotics HW3 - Part B: object detection (centroid + principal angle).

Usage:
    python3 hw3_b.py cubes.png
    python3 hw3_b.py cubes.png --method moments --no-show

Pipeline (binary machine vision, Lecture 6):
    1. grayscale -> threshold (triangle method, automatic) -> binary image
    2. morphology: opening removes specks, closing fills small gaps
    3. findContours -> one connected region R per object (holes filled)
    4. moments of R -> area m00, centroid (m10/m00, m01/m00)
    5. central moments -> principal angle  phi = 1/2 * atan2(2*mu11, mu20 - mu02)

Angle convention: image frame (x right, y down). Angles are measured from
the +x axis towards the +y axis, so a positive angle turns clockwise on screen.

Square objects: a square's second moments are the same in every direction
(mu20 == mu02, mu11 == 0 for any rotation), so phi is undefined for a cube's
top face and the value you get is set by noise. "auto" mode detects this
(eigenvalue ratio close to 1) and takes the angle from the edges of the
minimum-area bounding box instead, reported in (0, 90] because a square
looks the same after a 90 degree turn.

Other modules can reuse the detector:
    from hw3_b import detect_objects
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

# lambda_max / lambda_min below this -> region counts as "square-like"
# (1.5 ~ side ratio 1.22; a 2:1 rectangle gives 4.0)
SQUARE_RATIO = 1.5


# --------------------------------------------------------------------------
# 1-2. Segmentation
# --------------------------------------------------------------------------
def segment(img_bgr, thresh=None):
    """Return (threshold used, cleaned binary image)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if thresh is None:
        # Small objects on a large dark background: the histogram is one big
        # peak plus a long tail, so Otsu (which expects two peaks) fails here.
        # The triangle method is made for exactly this histogram shape.
        t, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        t, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # fill gaps
    return t, binary


# --------------------------------------------------------------------------
# 4-5. Moments, centroid, principal angle (Lecture 6 definitions)
# --------------------------------------------------------------------------
def region_moments(mask):
    """Moments of the region R = all non-zero pixels of mask.

    m_kj  = sum over R of x^k * y^j
    mu_kj = sum over R of (x - xc)^k * (y - yc)^j
    """
    ys, xs = np.nonzero(mask)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    m00 = float(x.size)                    # area A (pixels)
    m10, m01 = x.sum(), y.sum()
    xc, yc = m10 / m00, m01 / m00          # centroid

    dx, dy = x - xc, y - yc
    mu20 = float((dx * dx).sum())
    mu02 = float((dy * dy).sum())
    mu11 = float((dx * dy).sum())
    return m00, xc, yc, mu20, mu02, mu11


def principal_angle(mu20, mu02, mu11):
    """phi = 1/2 atan2(2 mu11, mu20 - mu02), returned in degrees, [0, 180)."""
    phi = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    return phi % 180.0


def eigen_ratio(mu20, mu02, mu11):
    """lambda_max / lambda_min of [[mu20, mu11], [mu11, mu02]].

    ~1 means the region is equally "heavy" in every direction (circle,
    square) and the principal axis is not defined.
    """
    mean = 0.5 * (mu20 + mu02)
    r = math.sqrt((0.5 * (mu20 - mu02)) ** 2 + mu11 ** 2)
    lmax, lmin = mean + r, mean - r
    return lmax / lmin if lmin > 1e-9 else float("inf")


def box_edge_angle(contour):
    """Direction of the longer edge of the minimum-area rectangle, [0, 180)."""
    pts = cv2.boxPoints(cv2.minAreaRect(contour))
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return math.degrees(math.atan2(float(e[1]), float(e[0]))) % 180.0


def fold_90(angle):
    """Map an angle to (0, 90] (a square repeats every 90 degrees)."""
    a = angle % 90.0
    return 90.0 if a < 1e-6 else a


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
def detect_objects(img_bgr, thresh=None, min_area=100, method="auto"):
    """Find objects and return (list of dicts, binary image, threshold).

    Each dict: id, centroid (x, y), angle (deg), angle_source,
               moment_angle, box_angle, area, eig_ratio, contour.
    """
    t, binary = segment(img_bgr, thresh)
    # [-2] works for OpenCV 3 (3 return values) and OpenCV 4/5 (2 values)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[-2]

    objects = []
    for c in contours:
        mask = np.zeros(binary.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
        m00, xc, yc, mu20, mu02, mu11 = region_moments(mask)
        if m00 < min_area:
            continue

        phi = principal_angle(mu20, mu02, mu11)
        ratio = eigen_ratio(mu20, mu02, mu11)
        box = box_edge_angle(c)

        if method == "moments" or (method == "auto" and ratio >= SQUARE_RATIO):
            angle, source = phi, "moments"
        elif method == "rect":
            angle, source = box, "box"
        else:  # auto, square-like region
            angle, source = fold_90(box), "box (square)"

        objects.append(dict(centroid=(xc, yc), angle=angle,
                            angle_source=source, moment_angle=phi,
                            box_angle=box, area=m00, eig_ratio=ratio,
                            contour=c))

    # stable IDs: bottom of the image first, then left to right
    objects.sort(key=lambda o: (-round(o["centroid"][1]), o["centroid"][0]))
    for i, o in enumerate(objects):
        o["id"] = i
    return objects, binary, t


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _put_text_deg(img, text, org, scale, color, thick=1, deg=False):
    """putText with an optional degree sign (Hershey fonts have no 'deg')."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)
    if deg:
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        r = max(2, int(round(3 * scale)))
        cv2.circle(img, (org[0] + w + r + 1, org[1] - h + r), r, color, 1,
                   cv2.LINE_AA)


def draw_result(img_bgr, objects, scale=2):
    """Annotated image: principal lines, centroids, IDs, plus a text header."""
    h0, w0 = img_bgr.shape[:2]
    canvas = cv2.resize(img_bgr, (w0 * scale, h0 * scale),
                        interpolation=cv2.INTER_NEAREST)
    H, W = canvas.shape[:2]
    L = 2 * (H + W)  # long enough to cross the whole image

    for o in objects:
        cx, cy = o["centroid"]
        # pixel centers sit at (i + 0.5) after the integer upscale
        px, py = (cx + 0.5) * scale, (cy + 0.5) * scale
        a = math.radians(o["angle"])
        p1 = (int(round(px - L * math.cos(a))), int(round(py - L * math.sin(a))))
        p2 = (int(round(px + L * math.cos(a))), int(round(py + L * math.sin(a))))
        ok, p1, p2 = cv2.clipLine((0, 0, W, H), p1, p2)
        cv2.drawContours(canvas, [o["contour"] * scale], -1, (0, 200, 0), 1,
                         offset=(scale // 2, scale // 2))
        if ok:
            cv2.line(canvas, p1, p2, (255, 60, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(px)), int(round(py))), 4, (0, 0, 255),
                   -1, cv2.LINE_AA)
        _put_text_deg(canvas, f"ID {o['id']}",
                      (int(px) + 14, int(py) - 14), 0.6, (255, 255, 255), 1)

    # white header with one line per object, like the sample output
    line_h, fs = 24, 0.55
    header = np.full((line_h * len(objects) + 16, W, 3), 255, np.uint8)
    for i, o in enumerate(objects):
        cx, cy = o["centroid"]
        txt = (f"ID {o['id']}: Centroid ({cx:.2f}, {cy:.2f}), "
               f"Angle {o['angle']:.2f}")
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        _put_text_deg(header, txt, ((W - tw) // 2 - 4, 22 + i * line_h), fs,
                      (0, 0, 0), 1, deg=True)
    return np.vstack([header, canvas])


def print_table(objects, t, method):
    print(f"threshold = {t:.0f} (grayscale), method = {method}, "
          f"{len(objects)} objects")
    print(f"{'ID':>2}  {'x_c':>8} {'y_c':>8}  {'angle':>7}  {'source':<13}"
          f"{'phi_mom':>8} {'area':>6} {'l1/l2':>6}")
    for o in objects:
        cx, cy = o["centroid"]
        print(f"{o['id']:>2}  {cx:8.2f} {cy:8.2f}  {o['angle']:7.2f}  "
              f"{o['angle_source']:<13}{o['moment_angle']:8.2f} "
              f"{o['area']:6.0f} {o['eig_ratio']:6.3f}")


# --------------------------------------------------------------------------
def hw3_b(path, thresh=None, method="auto", out=None, show=True,
          save_binary=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot read image '{path}'")
        return None

    objects, binary, t = detect_objects(img, thresh=thresh, method=method)
    print_table(objects, t, method)

    result = draw_result(img, objects)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = out or f"{stem}_b_result.png"
    cv2.imwrite(out, result)
    print(f"saved: {out}")
    if save_binary:
        bpath = f"{stem}_b_binary.png"
        cv2.imwrite(bpath, binary)
        print(f"saved: {bpath}")

    if show and sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("no display found: skipped the window (result PNG is saved)")
        show = False
    if show:
        try:
            cv2.imshow("Part B result (q / ESC to quit)", result)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("Part B result (q / ESC to quit)",
                                         cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        except (cv2.error, KeyboardInterrupt):
            pass  # no display available (e.g. SSH): the PNG is still saved
    return objects


def main():
    ap = argparse.ArgumentParser(description="HW3 Part B: centroid and "
                                 "principal angle of each object")
    ap.add_argument("image", help="input image, e.g. cubes.png")
    ap.add_argument("--thresh", type=int, default=None,
                    help="fixed gray threshold (default: automatic triangle)")
    ap.add_argument("--method", choices=["auto", "moments", "rect"],
                    default="auto",
                    help="angle source: auto (moments, box edge for squares),"
                         " moments only, or min-area box only")
    ap.add_argument("--out", default=None, help="output image path")
    ap.add_argument("--no-show", action="store_true",
                    help="do not open a window, just save the result")
    ap.add_argument("--save-binary", action="store_true",
                    help="also save the binary (thresholded) image")
    args = ap.parse_args()

    res = hw3_b(args.image, args.thresh, args.method, args.out,
                show=not args.no_show, save_binary=args.save_binary)
    sys.exit(0 if res is not None else 1)


if __name__ == "__main__":
    main()
