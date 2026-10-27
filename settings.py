"""
settings.py

I keep every tuned number in this one file so I don't have to hunt
through the code to change a threshold. The values were tuned on
project_video.mp4 and then checked on the two Udacity challenge videos.
"""

# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------
CALIBRATION_FOLDER = "camera_cal"
CALIBRATION_FILE = "camera_calibration.npz"
CHESSBOARD_CORNERS = (9, 6)  # inner corners per row and per column

# ---------------------------------------------------------------------------
# Perspective transform (bird's-eye view)
# ---------------------------------------------------------------------------
# "src": four points on the road in the camera image (a trapezoid around
#        our lane). They are mapped onto DST_POINTS, a rectangle in the
#        top-down image, so the lane lines come out parallel and vertical.
# "roi": region of interest, a wider trapezoid than "src" so the lines still
#        fit inside when the car drifts a bit or the road bends. Sky, trees
#        and the far side of the road are thrown away.
#
# Both depend on how the camera is mounted. challenge_video.mp4 was filmed
# with a different mount than project_video.mp4 (the lines meet at a point
# about 45 px further right and 13 px lower, and the lane looks narrower),
# so with one shared trapezoid its lines came out tilted and converging,
# and the "parallel lines" check kept rejecting them. That's why there are
# two profiles. I found the "challenge" points by mapping detected lines
# back into the camera image and intersecting them (vanishing point).
CAMERA_PROFILES = {
    # project_video.mp4 and harder_challenge_video.mp4
    "default": {
        "src": [[580, 460], [703, 460], [1060, 690], [250, 690]],
        "roi": [[530, 460], [755, 460], [1230, 700], [80, 700]],
    },
    # challenge_video.mp4
    "challenge": {
        "src": [[626, 470], [714, 470], [1005, 690], [330, 690]],
        "roi": [[580, 470], [770, 470], [1180, 700], [150, 700]],
    },
}
DST_POINTS = [[400, 0], [880, 0], [880, 720], [400, 720]]

# ---------------------------------------------------------------------------
# Colour and edge thresholds
# ---------------------------------------------------------------------------
# Yellow paint in HSV (OpenCV hue goes from 0 to 180).
YELLOW_LOW = (15, 80, 110)
YELLOW_HIGH = (35, 255, 255)

# Bright stripe test (works for white AND yellow paint).
# History: a fixed brightness threshold flooded on the light concrete
# bridges. "Brighter than the local average" was better, but it also fired
# on the sunlit side of concrete barriers and shadow borders. What finally
# worked: paint is a thin stripe, so it is brighter than the road a few
# pixels to its LEFT and to its RIGHT at the same time. A barrier or shadow
# border is only brighter on one side.
STRIPE_DISTANCE = 12      # how far left/right (pixels) we compare
STRIPE_CONTRAST = 15      # how much brighter than both sides
STRIPE_MIN_L = 110        # and not a dark pixel overall

# Sobel edge in x direction (lane lines are mostly vertical in the image).
# On its own it fired on every shadow and seam, so an edge pixel only
# counts if it also passes a much weaker stripe test. This helps a bit
# with faded paint that just misses the normal stripe test.
SOBEL_KERNEL = 3
SOBEL_MIN = 25
WEAK_STRIPE_CONTRAST = 5

# ---------------------------------------------------------------------------
# Lane pixel search in the bird's-eye image
# ---------------------------------------------------------------------------
N_WINDOWS = 9          # number of stacked sliding windows per line
WINDOW_MARGIN = 70     # half width of a window in pixels
WINDOW_MIN_PIXELS = 40 # pixels needed before a window re-centres itself
AROUND_FIT_MARGIN = 60 # half width of the band used when we already know the line
MIN_LINE_PIXELS = 300  # fewer pixels than this and I don't even try to fit

# ---------------------------------------------------------------------------
# Plausibility checks (in bird's-eye pixels)
# ---------------------------------------------------------------------------
EXPECTED_LANE_WIDTH = 480         # distance between DST_POINTS columns
MIN_LANE_WIDTH = 330
MAX_LANE_WIDTH = 640
MAX_WIDTH_CHANGE_TOP_BOTTOM = 160 # lines must be roughly parallel
MAX_JUMP_FROM_HISTORY = 70        # max shift of a line between frames

# ---------------------------------------------------------------------------
# Confidence score and temporal smoothing
# ---------------------------------------------------------------------------
# The score is the average of four parts, each between 0 and 1.
# It is a heuristic quality score, NOT a calibrated probability.
PIXELS_FOR_FULL_SCORE = 2000   # this many supporting pixels gives full marks
COVERAGE_FOR_FULL_SCORE = 0.7  # pixels spread over 70 % of the height
RMS_FOR_ZERO_SCORE = 15        # fit error (px) at which the fit part hits 0

GOOD_SCORE = 0.60   # at or above this a boundary is "DETECTED"
REJECT_SCORE = 0.35  # below this the new fit is thrown away
SMOOTHING_FRAMES = 6 # how many accepted fits are averaged
MAX_HOLD_FRAMES = 12 # how long (about 0.5 s at 25 fps) a lost line is held

# ---------------------------------------------------------------------------
# Vehicle position and curvature
# ---------------------------------------------------------------------------
# Offset is measured as a fraction of the lane width.
# 0 = centred, -0.5 / +0.5 = car centre sits on the left / right line.
CENTRED_LIMIT = 0.08
CLOSE_TO_LINE_LIMIT = 0.20

# Rough conversion to metres, used only for the curve radius. It assumes
# a 3.7 m lane is EXPECTED_LANE_WIDTH pixels wide and the warped image
# shows about 30 m of road. It is not a calibrated value.
METRES_PER_PIXEL_X = 3.7 / EXPECTED_LANE_WIDTH
METRES_PER_PIXEL_Y = 30.0 / 720
STRAIGHT_RADIUS = 2500  # above this the road is shown as "nearly straight"
