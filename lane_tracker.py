"""
lane_tracker.py

Takes the raw per-frame detections from lane_search.py and decides what we
actually believe. This is where:

  * each boundary gets a confidence score,
  * implausible results get rejected (crossing lines, weird lane width,
    a line that suddenly jumps sideways),
  * accepted fits are averaged over the last few frames (smoothing),
  * a lost boundary is held for a short time and then dropped.

Each boundary ends up in one of four states:
  DETECTED   - fresh detection this frame with a good score
  UNCERTAIN  - fresh detection, accepted, but with a low score
  HOLDING    - nothing usable this frame, we show the old position
               (drawn dashed and greyed so it does not look like a new detection)
  LOST       - held for too long, nothing is shown
"""

from collections import deque

import numpy as np

import settings

IMAGE_HEIGHT = 720
# y values (rows of the bird's-eye image) where two curves get compared
CHECK_ROWS = np.linspace(0, IMAGE_HEIGHT - 1, 10)


def x_at(fit, y):
    return fit[0] * y ** 2 + fit[1] * y + fit[2]


def average_distance(fit_a, fit_b):
    """Mean horizontal distance in pixels between two curves."""
    return float(np.mean(np.abs(x_at(fit_a, CHECK_ROWS) - x_at(fit_b, CHECK_ROWS))))


def score_detection(detection, history_fit):
    """
    Confidence score between 0 and 1 for one boundary, built from:
      - how many pixels support the fit
      - how much of the image height they cover
      - how well the curve goes through them
      - how close it is to what we saw in recent frames
    Every part is 0..1 and the score is their plain average.
    """
    pixel_part = min(1.0, detection["n_pixels"] / settings.PIXELS_FOR_FULL_SCORE)
    coverage_part = min(1.0, detection["coverage"] / settings.COVERAGE_FOR_FULL_SCORE)
    fit_part = max(0.0, 1.0 - detection["rms_error"] / settings.RMS_FOR_ZERO_SCORE)

    if history_fit is None:
        consistency_part = 0.5  # nothing to compare with yet, so neutral
    else:
        shift = average_distance(detection["fit"], history_fit)
        consistency_part = max(0.0, 1.0 - shift / settings.MAX_JUMP_FROM_HISTORY)

    parts = {
        "pixels": pixel_part,
        "coverage": coverage_part,
        "fit": fit_part,
        "consistency": consistency_part,
    }
    return float(np.mean(list(parts.values()))), parts


def pair_is_plausible(left_fit, right_fit):
    """
    Do these two curves look like the two sides of one lane?
    Returns (True, "") or (False, reason).
    """
    widths = x_at(right_fit, CHECK_ROWS) - x_at(left_fit, CHECK_ROWS)
    bottom_width = widths[-1]

    if np.any(widths <= 0):
        return False, "lines cross"
    if not settings.MIN_LANE_WIDTH <= bottom_width <= settings.MAX_LANE_WIDTH:
        return False, "lane width"
    if widths.max() - widths.min() > settings.MAX_WIDTH_CHANGE_TOP_BOTTOM:
        return False, "not parallel"
    return True, ""


class Boundary:
    """Everything we remember about one lane boundary (left or right)."""

    def __init__(self, name):
        self.name = name
        self.recent_fits = deque(maxlen=settings.SMOOTHING_FRAMES)
        self.smoothed_fit = None
        self.status = "LOST"
        self.score = 0.0
        self.score_parts = {}
        self.frames_without_detection = 0
        self.note = ""          # short reason shown on screen
        self.last_detection = None

    def accept(self, fit, score):
        self.recent_fits.append(fit)
        # Smoothing: simply average the coefficients of the last few fits.
        self.smoothed_fit = np.mean(self.recent_fits, axis=0)
        self.score = score
        self.frames_without_detection = 0
        self.status = "DETECTED" if score >= settings.GOOD_SCORE else "UNCERTAIN"

    def miss(self, reason):
        self.note = reason
        self.frames_without_detection += 1
        # The score shown while holding fades out, because the old position
        # gets less trustworthy with every frame that passes.
        self.score *= 0.8

        if self.smoothed_fit is not None and \
                self.frames_without_detection <= settings.MAX_HOLD_FRAMES:
            self.status = "HOLDING"
        else:
            # Held too long: forget everything, next search starts fresh.
            self.status = "LOST"
            self.smoothed_fit = None
            self.recent_fits.clear()
            self.score = 0.0

    @property
    def visible(self):
        return self.status != "LOST"

    @property
    def fresh(self):
        """True only for a real new detection in this frame."""
        return self.status in ("DETECTED", "UNCERTAIN")


class LaneTracker:
    def __init__(self):
        self.left = Boundary("left")
        self.right = Boundary("right")

    def update(self, left_detection, right_detection):
        candidates = {}  # side name -> (fit, score) that survived the checks

        # Step 1: check each side on its own.
        for boundary, detection in ((self.left, left_detection),
                                    (self.right, right_detection)):
            boundary.last_detection = detection
            boundary.note = ""
            if detection is None:
                boundary.note = "too few pixels"
                continue

            score, parts = score_detection(detection, boundary.smoothed_fit)
            boundary.score_parts = parts

            if boundary.smoothed_fit is not None and \
                    average_distance(detection["fit"], boundary.smoothed_fit) > \
                    settings.MAX_JUMP_FROM_HISTORY:
                boundary.note = "jumped"
                continue
            if score < settings.REJECT_SCORE:
                boundary.note = "low score"
                continue
            candidates[boundary.name] = (detection["fit"], score)

        # Step 2: check the two sides against each other. If one side is
        # missing this frame, compare the other one with its held position.
        left_fit = candidates.get("left", (self.left.smoothed_fit,))[0]
        right_fit = candidates.get("right", (self.right.smoothed_fit,))[0]
        if candidates and left_fit is not None and right_fit is not None:
            ok, reason = pair_is_plausible(left_fit, right_fit)
            if not ok:
                # Throw away the weaker new candidate. If there is only one
                # candidate and it doesn't fit with the other side's held
                # line, it is dropped too. That is what stops the overlay
                # from jumping onto a neighbouring marking.
                weakest = min(candidates, key=lambda side: candidates[side][1])
                del candidates[weakest]
                getattr(self, weakest).note = reason

        # Step 3: update both boundaries.
        for boundary in (self.left, self.right):
            if boundary.name in candidates:
                fit, score = candidates[boundary.name]
                boundary.accept(fit, score)
            else:
                boundary.miss(boundary.note or "rejected")


def lane_position(left_fit, right_fit, image_width=1280, image_height=720):
    """
    Where is the car inside the lane?

    The camera sits in the middle of the car, and the bird's-eye view is
    symmetric around the image centre, so the car centre is the middle
    column at the bottom of the image.

    Returns the offset as a fraction of the lane width:
      0     -> perfectly centred
      < 0   -> car is left of the lane centre
      > 0   -> car is right of the lane centre
      +-0.5 -> car centre is on a lane line
    """
    y = image_height - 1
    left_x = x_at(left_fit, y)
    right_x = x_at(right_fit, y)
    lane_width = right_x - left_x
    lane_centre = (left_x + right_x) / 2
    car_centre = image_width / 2
    return (car_centre - lane_centre) / lane_width


def position_text(offset):
    side = "left" if offset < 0 else "right"
    size = abs(offset)
    if size < settings.CENTRED_LIMIT:
        return "CENTRED"
    if size < settings.CLOSE_TO_LINE_LIMIT:
        return f"SHIFTED {side.upper()}"
    return f"CLOSE TO {side.upper()} LINE"


def curve_radius(left_fit, right_fit, image_height=720):
    """
    Rough curve radius in metres, using the lane-centre curve.
    Radius of x(y) is (1 + x'^2)^1.5 / |x''|. Before applying it I convert
    the coefficients from pixels to metres with the rough scale in settings.
    Returns (radius, "left"/"right").
    """
    centre_fit = (np.array(left_fit) + np.array(right_fit)) / 2
    mx, my = settings.METRES_PER_PIXEL_X, settings.METRES_PER_PIXEL_Y
    a = centre_fit[0] * mx / my ** 2
    b = centre_fit[1] * mx / my
    y = image_height * my  # evaluate at the car, in metres

    if abs(a) < 1e-9:
        return float("inf"), "straight"
    radius = (1 + (2 * a * y + b) ** 2) ** 1.5 / abs(2 * a)
    # a > 0 means the curve moves to the right as we look further ahead
    # (smaller y), so the road bends right.
    direction = "right" if a > 0 else "left"
    return radius, direction
