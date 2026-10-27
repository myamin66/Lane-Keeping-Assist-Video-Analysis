"""
overlay.py

Everything that is drawn on the output video:
  * left boundary (blue) and right boundary (orange), each drawn
      solid   when DETECTED,
      thinner when UNCERTAIN,
      dashed and greyed when HOLDING (old position, not a new detection),
      not at all when LOST
  * the lane area, the lane centre (dashed white) and the car centre
  * an info panel with status + score per boundary, the car position and
    a rough curve radius
  * a small bird's-eye picture of the lane search in the corner
"""

import cv2
import numpy as np

import settings
from lane_tracker import curve_radius, lane_position, position_text, x_at

# Colours are BGR in OpenCV.
LEFT_COLOUR = (255, 128, 0)    # blue
RIGHT_COLOUR = (0, 140, 255)   # orange
STATUS_COLOURS = {
    "DETECTED": (80, 220, 80),
    "UNCERTAIN": (0, 220, 255),
    "HOLDING": (0, 140, 255),
    "LOST": (60, 60, 255),
}
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def curve_in_camera_view(fit, camera):
    """Points of a bird's-eye curve, moved back into the camera image."""
    ys = np.linspace(0, 719, 40)
    xs = x_at(fit, ys)
    points = np.float32(np.column_stack([xs, ys])).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, camera.from_birdseye).reshape(-1, 2)


def greyed(colour):
    """Mix a colour with grey so held lines look 'faded'."""
    return tuple(int(0.4 * c + 0.6 * 140) for c in colour)


def draw_dashed(image, points, colour, thickness):
    points = np.int32(points)
    for i in range(0, len(points) - 1, 2):  # draw every other segment
        cv2.line(image, tuple(points[i]), tuple(points[i + 1]), colour, thickness)


def draw_boundary(image, boundary, colour, camera):
    if not boundary.visible:
        return
    points = curve_in_camera_view(boundary.smoothed_fit, camera)
    if boundary.status == "DETECTED":
        cv2.polylines(image, [np.int32(points)], False, colour, 10)
    elif boundary.status == "UNCERTAIN":
        cv2.polylines(image, [np.int32(points)], False, colour, 5)
    else:  # HOLDING
        draw_dashed(image, points, greyed(colour), 6)


def draw_lane(image, tracker, camera):
    left, right = tracker.left, tracker.right
    draw_boundary(image, left, LEFT_COLOUR, camera)
    draw_boundary(image, right, RIGHT_COLOUR, camera)
    if not (left.visible and right.visible):
        return

    # Lane area: green when both sides are fresh and good, amber otherwise.
    left_points = curve_in_camera_view(left.smoothed_fit, camera)
    right_points = curve_in_camera_view(right.smoothed_fit, camera)
    polygon = np.int32(np.vstack([left_points, right_points[::-1]]))
    both_good = left.status == "DETECTED" and right.status == "DETECTED"
    fill_colour = (0, 180, 0) if both_good else (0, 170, 255)
    layer = image.copy()
    cv2.fillPoly(layer, [polygon], fill_colour)
    cv2.addWeighted(layer, 0.25, image, 0.75, 0, dst=image)

    # Lane centre line, dashed white.
    centre_fit = (left.smoothed_fit + right.smoothed_fit) / 2
    draw_dashed(image, curve_in_camera_view(centre_fit, camera), WHITE, 3)

    # Car centre (the camera sits in the middle of the car) vs lane centre,
    # both marked at the bottom of the image.
    h, w = image.shape[:2]
    lane_centre_x = int(curve_in_camera_view(centre_fit, camera)[-1][0])
    car_x = w // 2
    cv2.line(image, (car_x, h - 70), (car_x, h - 10), (0, 255, 255), 4)
    cv2.circle(image, (lane_centre_x, h - 40), 8, WHITE, -1)
    cv2.arrowedLine(image, (lane_centre_x, h - 40), (car_x, h - 40),
                    (0, 255, 255), 2, tipLength=0.3)


def put_text(image, text, position, colour=WHITE, scale=0.6, thickness=2):
    cv2.putText(image, text, position, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_score_bar(image, x, y, score, colour):
    cv2.rectangle(image, (x, y), (x + 120, y + 14), (90, 90, 90), -1)
    cv2.rectangle(image, (x, y), (x + int(120 * score), y + 14), colour, -1)
    cv2.rectangle(image, (x, y), (x + 120, y + 14), WHITE, 1)


def draw_boundary_row(image, boundary, y, colour):
    status = boundary.status
    if status == "HOLDING":
        status += f" ({boundary.frames_without_detection})"
    put_text(image, boundary.name.upper(), (15, y), colour, 0.65)
    put_text(image, status, (95, y), STATUS_COLOURS[boundary.status], 0.6)
    put_text(image, f"score {boundary.score:.2f}", (290, y), WHITE, 0.55, 1)
    draw_score_bar(image, 400, y - 13, boundary.score, STATUS_COLOURS[boundary.status])
    if not boundary.fresh and boundary.note:
        put_text(image, f"({boundary.note})", (535, y), (200, 200, 200), 0.5, 1)


def draw_position_bar(image, x, y, offset):
    """A small lane picture: two lines, the centre tick and the car."""
    width = 200
    cv2.line(image, (x, y - 12), (x, y + 12), LEFT_COLOUR, 3)
    cv2.line(image, (x + width, y - 12), (x + width, y + 12), RIGHT_COLOUR, 3)
    cv2.line(image, (x, y), (x + width, y), (150, 150, 150), 1)
    cv2.line(image, (x + width // 2, y - 6), (x + width // 2, y + 6), WHITE, 1)
    car_x = int(x + width / 2 + offset * width)
    car_x = max(x, min(x + width, car_x))
    cv2.rectangle(image, (car_x - 6, y - 9), (car_x + 6, y + 9), (0, 255, 255), -1)


def draw_info_panel(image, tracker):
    left, right = tracker.left, tracker.right

    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (930, 150), (30, 30, 30), -1)
    cv2.addWeighted(panel, 0.65, image, 0.35, 0, dst=image)

    draw_boundary_row(image, left, 30, LEFT_COLOUR)
    draw_boundary_row(image, right, 60, RIGHT_COLOUR)

    # Position needs both boundaries. If one of them is only held from
    # earlier frames, the position is shown but marked as uncertain.
    if left.visible and right.visible:
        offset = lane_position(left.smoothed_fit, right.smoothed_fit)
        text = position_text(offset)
        colour = (60, 60, 255) if text.startswith("CLOSE") else WHITE
        line = f"POSITION: {text}  ({offset * 100:+.0f}% of lane width)"
        if not (left.fresh and right.fresh):
            line += "  - uncertain"
            colour = (0, 170, 255)
        put_text(image, line, (15, 95), colour, 0.6)
        draw_position_bar(image, 700, 90, offset)

        radius, direction = curve_radius(left.smoothed_fit, right.smoothed_fit)
        if radius > settings.STRAIGHT_RADIUS:
            curve_text = "CURVE: nearly straight"
        else:
            curve_text = f"CURVE: radius ~{radius:.0f} m to the {direction} (rough estimate)"
        put_text(image, curve_text, (15, 130), (200, 200, 200), 0.55, 1)
    else:
        put_text(image, "POSITION: unknown - a lane boundary is missing",
                 (15, 95), (60, 60, 255), 0.6)


def birdseye_search_view(binary_warped, tracker):
    """
    Bird's-eye picture of the search: mask in grey, left pixels in blue,
    right pixels in orange, sliding windows in green, curves in yellow.
    """
    view = cv2.cvtColor(binary_warped // 3, cv2.COLOR_GRAY2BGR)
    for boundary, colour in ((tracker.left, LEFT_COLOUR), (tracker.right, RIGHT_COLOUR)):
        detection = boundary.last_detection
        if detection is not None:
            view[detection["ys"], detection["xs"]] = colour
            for (x1, y1, x2, y2) in detection["windows"]:
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if boundary.visible:
            ys = np.linspace(0, 719, 40)
            points = np.int32(np.column_stack([x_at(boundary.smoothed_fit, ys), ys]))
            if boundary.fresh:
                cv2.polylines(view, [points], False, (0, 255, 255), 4)
            else:
                draw_dashed(view, points, (0, 255, 255), 4)
    return view


def draw_overlay(frame, tracker, camera, binary_warped):
    output = frame.copy()
    draw_lane(output, tracker, camera)
    draw_info_panel(output, tracker)

    # Small bird's-eye inset in the top-right corner.
    inset = cv2.resize(birdseye_search_view(binary_warped, tracker), (320, 180))
    output[10:190, 950:1270] = inset
    cv2.rectangle(output, (950, 10), (1270, 190), WHITE, 1)
    put_text(output, "bird's-eye search", (958, 30), WHITE, 0.5, 1)
    return output
