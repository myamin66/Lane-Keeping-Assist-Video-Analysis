"""
lane_search.py

Finds the pixels of one lane boundary in the bird's-eye mask and fits a
curve through them. Two ways to search:

  * sliding windows: start at the bottom of the image and climb upwards in
    small boxes, re-centring each box on the pixels it finds. Used when we
    don't know where the line is (first frame, or after losing it).

  * around the previous fit: if we already know roughly where the line is,
    just take the pixels in a band around last frame's curve. This is faster
    and much less likely to jump onto a different marking.

The curve is a 2nd order polynomial x = a*y^2 + b*y + c. I write x as a
function of y because lane lines are close to vertical in the bird's-eye
image, so one y value has exactly one x value.
"""

import numpy as np

import settings


def histogram_start(binary_warped, side):
    """
    Guess where a line starts at the bottom of the image by counting
    white pixels per column in the lower half. I only look on our own
    side of the image centre and not too far out, so we don't grab the
    next lane's marking.
    """
    h, w = binary_warped.shape
    histogram = np.sum(binary_warped[h // 2:, :] > 0, axis=0)
    centre = w // 2
    reach = settings.EXPECTED_LANE_WIDTH  # never look more than a lane away

    if side == "left":
        start, stop = centre - reach, centre
    else:
        start, stop = centre, centre + reach

    part = histogram[start:stop]
    if part.max() == 0:
        return None
    return start + int(np.argmax(part))


def sliding_window_search(binary_warped, start_x):
    """Climb up from start_x with stacked windows, collect pixels."""
    h, _ = binary_warped.shape
    ys_all, xs_all = binary_warped.nonzero()

    window_height = h // settings.N_WINDOWS
    margin = settings.WINDOW_MARGIN
    current_x = start_x

    picked = []
    windows = []  # kept only so the diagnostic view can draw them

    for i in range(settings.N_WINDOWS):
        y_low = h - (i + 1) * window_height
        y_high = h - i * window_height
        x_low = current_x - margin
        x_high = current_x + margin
        windows.append((x_low, y_low, x_high, y_high))

        inside = ((ys_all >= y_low) & (ys_all < y_high) &
                  (xs_all >= x_low) & (xs_all < x_high)).nonzero()[0]
        picked.append(inside)

        # Enough pixels? Then move the next window to their average x.
        # Otherwise (e.g. a gap in a dashed line) keep going straight up.
        if len(inside) > settings.WINDOW_MIN_PIXELS:
            current_x = int(np.mean(xs_all[inside]))

    picked = np.concatenate(picked)
    return xs_all[picked], ys_all[picked], windows


def search_around_fit(binary_warped, fit):
    """Take all pixels within a band around an existing curve."""
    ys_all, xs_all = binary_warped.nonzero()
    curve_x = fit[0] * ys_all ** 2 + fit[1] * ys_all + fit[2]
    inside = np.abs(xs_all - curve_x) < settings.AROUND_FIT_MARGIN
    return xs_all[inside], ys_all[inside]


def find_boundary(binary_warped, side, previous_fit):
    """
    Search for one boundary ("left" or "right") and fit a curve.

    Returns a dict with the fit and some numbers the tracker uses to score
    it, or None if there were not enough pixels to fit anything.
    """
    h = binary_warped.shape[0]
    windows = []
    method = "around previous"
    xs, ys = np.array([]), np.array([])

    if previous_fit is not None:
        xs, ys = search_around_fit(binary_warped, previous_fit)

    if len(xs) < settings.MIN_LINE_PIXELS:
        # Either no history or the band was empty: go back to windows.
        # If we have a previous fit, start from its bottom point rather than
        # the histogram, so we stay on the same marking.
        method = "sliding windows"
        if previous_fit is not None:
            start_x = int(np.polyval(previous_fit, h - 1))
        else:
            start_x = histogram_start(binary_warped, side)
        if start_x is None:
            return None
        xs, ys, windows = sliding_window_search(binary_warped, start_x)

    if len(xs) < settings.MIN_LINE_PIXELS:
        return None

    fit = np.polyfit(ys, xs, 2)

    # How well does the curve go through its own pixels? Root mean square
    # distance in pixels: small = clean line, large = scattered noise.
    residuals = xs - np.polyval(fit, ys)
    rms_error = float(np.sqrt(np.mean(residuals ** 2)))

    # How much of the image height the pixels cover. A line found only in
    # the bottom 10 % gives a very unreliable curve further up.
    vertical_coverage = float((ys.max() - ys.min()) / h)

    return {
        "fit": fit,
        "xs": xs,
        "ys": ys,
        "n_pixels": len(xs),
        "rms_error": rms_error,
        "coverage": vertical_coverage,
        "windows": windows,
        "method": method,
    }
