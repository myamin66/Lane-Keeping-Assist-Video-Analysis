"""
calibrate_camera.py

Run this once. It looks at the chessboard photos in camera_cal/, finds the
corners, and works out how the camera lens bends the image. The result is
saved to camera_calibration.npz so the main script can undistort frames.

    python calibrate_camera.py
"""

import glob
import os

import cv2
import numpy as np

import settings


def main():
    cols, rows = settings.CHESSBOARD_CORNERS

    # Where the corners are on a flat chessboard: (0,0,0), (1,0,0), ... (8,5,0).
    # The real square size does not matter for undistortion.
    board_points = np.zeros((rows * cols, 3), np.float32)
    board_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    object_points = []  # 3D points on the board
    image_points = []   # matching 2D points in the photo
    image_size = None

    photos = sorted(glob.glob(os.path.join(settings.CALIBRATION_FOLDER, "*.jpg")))
    print(f"Found {len(photos)} calibration photos")

    for path in photos:
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            object_points.append(board_points)
            image_points.append(corners)
        else:
            # A few photos crop part of the board, that's fine, we skip them.
            print(f"  corners not found in {os.path.basename(path)}, skipping")

    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)

    np.savez(settings.CALIBRATION_FILE,
             camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Used {len(object_points)} photos, saved {settings.CALIBRATION_FILE}")

    # Save a before/after picture so we can check it worked.
    os.makedirs("screenshots", exist_ok=True)
    test = cv2.imread(photos[0])
    fixed = cv2.undistort(test, camera_matrix, dist_coeffs)
    cv2.imwrite(os.path.join("screenshots", "00_calibration_check.jpg"),
                np.hstack([test, fixed]))


if __name__ == "__main__":
    main()
