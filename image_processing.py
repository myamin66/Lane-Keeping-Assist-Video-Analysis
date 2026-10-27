"""
image_processing.py

The per-frame image steps that come before the lane search:
  1. undistort the frame with the camera calibration
  2. build a colour mask (yellow paint + thin bright stripes)
  3. build an edge mask (Sobel in x direction, only on weak stripes)
  4. combine both into one black/white "lane pixel" image
  5. warp it to a bird's-eye view
"""

import cv2
import numpy as np

import settings


class Camera:
    """Holds the calibration, the perspective matrices and the ROI."""

    def __init__(self, profile="default"):
        data = np.load(settings.CALIBRATION_FILE)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        self.src_points = settings.CAMERA_PROFILES[profile]["src"]
        self.roi_points = settings.CAMERA_PROFILES[profile]["roi"]

        src = np.float32(self.src_points)
        dst = np.float32(settings.DST_POINTS)
        self.to_birdseye = cv2.getPerspectiveTransform(src, dst)
        self.from_birdseye = cv2.getPerspectiveTransform(dst, src)

    def undistort(self, frame):
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def warp(self, image):
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, self.to_birdseye, (w, h),
                                   flags=cv2.INTER_LINEAR)


def region_of_interest_mask(shape, roi_points):
    """White inside the trapezoid we look at, black everywhere else."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(roi_points)], 255)
    return mask


def stripe_mask(lightness, contrast):
    """
    Pixels that are brighter than the road STRIPE_DISTANCE pixels to the
    left AND to the right of them, i.e. thin bright stripes like paint.
    np.roll shifts the whole image sideways so we can compare every pixel
    with its neighbours in one go. (It wraps around at the image border,
    but the border is outside the ROI anyway.)
    """
    d = settings.STRIPE_DISTANCE
    centre = lightness.astype(np.int16)  # int16 so differences can be negative
    left_neighbour = np.roll(centre, d, axis=1)
    right_neighbour = np.roll(centre, -d, axis=1)
    stripe = ((centre - left_neighbour) > contrast) &              ((centre - right_neighbour) > contrast) &              (centre >= settings.STRIPE_MIN_L)
    return np.where(stripe, 255, 0).astype(np.uint8)


def colour_mask(frame):
    """Pixels that look like lane paint: yellow, or a bright thin stripe."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]

    yellow = cv2.inRange(hsv, settings.YELLOW_LOW, settings.YELLOW_HIGH)
    bright_stripes = stripe_mask(lightness, settings.STRIPE_CONTRAST)
    return cv2.bitwise_or(yellow, bright_stripes)


def edge_mask(frame):
    """Strong Sobel-x edges that also sit on a (weak) bright stripe."""
    lightness = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]

    sobel_x = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=settings.SOBEL_KERNEL)
    sobel_x = np.absolute(sobel_x)
    # Scale to 0..255 so the threshold does not depend on the frame contrast.
    scaled = np.uint8(255 * sobel_x / max(sobel_x.max(), 1))
    strong = np.where(scaled >= settings.SOBEL_MIN, 255, 0).astype(np.uint8)

    weak_stripes = stripe_mask(lightness, settings.WEAK_STRIPE_CONTRAST)
    return cv2.bitwise_and(strong, weak_stripes)


def lane_pixel_mask(frame, roi_points):
    """
    Combine colour and edges and cut away everything outside the ROI.
    Returns all three masks because the diagnostic view shows them.
    """
    colour = colour_mask(frame)
    edges = edge_mask(frame)
    combined = cv2.bitwise_or(colour, edges)
    combined = cv2.bitwise_and(combined, region_of_interest_mask(frame.shape, roi_points))
    return colour, edges, combined
