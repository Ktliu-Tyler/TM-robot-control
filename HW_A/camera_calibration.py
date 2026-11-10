import numpy as np
import cv2 as cv
import glob
import os

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob("./Robotics_Image/*.jpg")

# Find chessboard corners in all images
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    if h > w:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)

    # If found, add object points and image points
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, (8, 6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(2500)

cv.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez("part_a_camera.npz", K=mtx, D=dist, size=gray.shape[::-1])
os.makedirs("Cal_Results", exist_ok=True)

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Undistortion
for fname in images:
    img = cv.imread(fname)
    h, w = img.shape[:2]
    if h > w:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    
    h, w = img.shape[:2]

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h)
    )

    # Using cv.undistort()
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image
    x, y, roi_w, roi_h = roi
    dst = dst[y:y+roi_h, x:x+roi_w]

    filename = os.path.basename(fname)
    cv.imwrite(f'./Cal_Undistort/calibresult_{filename}', dst)

    # Using Remapping
    mapx, mapy = cv.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5
    )
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # Crop the image
    dst = dst[y:y+roi_h, x:x+roi_w]

    cv.imwrite(f'./Cal_Remapping/calibresult_{filename}', dst)

# Undistored random things
img = cv.imread("./Image_21.jpg")

# Match the landscape orientation used for calibration
h, w = img.shape[:2]
if h > w:
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

h, w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 1, (w, h)
)

dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# Crop the image
x, y, roi_w, roi_h = roi
dst = dst[y:y+roi_h, x:x+roi_w]

cv.imwrite("./Cal_Results/calibresult_21.jpg", dst)

# Reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    error = cv.norm(
        imgpoints[i].reshape(-1, 2),
        imgpoints2.reshape(-1, 2).astype(np.float32),
        cv.NORM_L2
    ) / len(imgpoints2)
    mean_error += error

print("Total error: {}".format(mean_error / len(objpoints)))
print("RMS reprojection error:", ret)
