"""
pipeline.py

Puts all the steps together for one video frame:

  frame -> undistort -> colour + edge masks -> ROI -> bird's-eye warp
        -> search left / right boundary -> tracker (checks, smoothing)
        -> overlay

It also keeps the intermediate images, so we can build the diagnostic
video and the pipeline screenshots from the same code.
"""

import cv2
import numpy as np

from image_processing import Camera, lane_pixel_mask
from lane_search import find_boundary
from lane_tracker import LaneTracker
from overlay import birdseye_search_view, draw_overlay, put_text


class LaneKeepingAssist:
    def __init__(self, camera_profile="default"):
        self.camera = Camera(camera_profile)
        self.tracker = LaneTracker()

    def process(self, frame):
        """Returns the annotated frame and a dict with the in-between images."""
        # Everything is tuned for 1280x720, so other videos are resized first.
        if frame.shape[:2] != (720, 1280):
            frame = cv2.resize(frame, (1280, 720))

        undistorted = self.camera.undistort(frame)
        colour, edges, combined = lane_pixel_mask(undistorted, self.camera.roi_points)
        binary_warped = self.camera.warp(combined)

        # Search near last frame's smoothed line when we still have one.
        left = find_boundary(binary_warped, "left", self.tracker.left.smoothed_fit)
        right = find_boundary(binary_warped, "right", self.tracker.right.smoothed_fit)
        self.tracker.update(left, right)

        annotated = draw_overlay(undistorted, self.tracker, self.camera, binary_warped)

        stages = {
            "undistorted": undistorted,
            "roi": roi_view(undistorted, self.camera),
            "colour": colour,
            "edges": edges,
            "combined": combined,
            "warped": self.camera.warp(undistorted),
            "binary_warped": binary_warped,
            "search": birdseye_search_view(binary_warped, self.tracker),
            "final": annotated,
        }
        return annotated, stages


def roi_view(undistorted, camera):
    """Undistorted frame with the ROI (green) and the warp trapezoid (red)."""
    view = undistorted.copy()
    cv2.polylines(view, [np.int32(camera.roi_points)], True, (0, 255, 0), 3)
    cv2.polylines(view, [np.int32(camera.src_points)], True, (0, 0, 255), 2)
    return view


def diagnostic_grid(stages):
    """Six stages side by side in a 3 x 2 grid (1920 x 720 pixels)."""
    def tile(image, title):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (640, 360))
        # Label at the bottom, so it doesn't hide the info panel in tile 6.
        cv2.rectangle(image, (0, 328), (640, 360), (0, 0, 0), -1)
        put_text(image, title, (10, 351), (255, 255, 255), 0.65)
        return image

    top = np.hstack([
        tile(stages["roi"], "1. undistorted + ROI (green) + warp area (red)"),
        tile(stages["colour"], "2. colour mask (yellow + bright stripes)"),
        tile(stages["edges"], "3. Sobel-x edges on weak stripes"),
    ])
    bottom = np.hstack([
        tile(stages["combined"], "4. combined mask inside ROI"),
        tile(stages["search"], "5. bird's-eye + lane pixel search"),
        tile(stages["final"], "6. final overlay"),
    ])
    return np.vstack([top, bottom])
