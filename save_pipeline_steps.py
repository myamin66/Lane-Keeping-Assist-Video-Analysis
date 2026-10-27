"""
save_pipeline_steps.py

Saves one picture per processing stage for a single moment of a video,
for the visual walkthrough of the pipeline.

    python save_pipeline_steps.py videos/project_video.mp4 --time 22
    python save_pipeline_steps.py videos/challenge_video.mp4 --time 5 --camera challenge

The frames before --time are processed too (without saving), so the tracker
has its normal history and the picture matches what the video shows.
Pictures go to screenshots/<video name>_<time>s/.
"""

import argparse
import os

import cv2

from pipeline import LaneKeepingAssist, diagnostic_grid


def main():
    parser = argparse.ArgumentParser(description="Save the pipeline stages of one frame")
    parser.add_argument("video")
    parser.add_argument("--time", type=float, required=True, help="moment in seconds")
    parser.add_argument("--camera", default="default", choices=["default", "challenge"])
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    target_frame = int(args.time * fps)

    lka = LaneKeepingAssist(args.camera)
    stages = None
    for _ in range(target_frame + 1):
        ok, frame = video.read()
        if not ok:
            break
        _, stages = lka.process(frame)
    if stages is None:
        raise SystemExit("Could not read the video")

    name = os.path.splitext(os.path.basename(args.video))[0]
    folder = os.path.join("screenshots", f"{name}_{args.time:g}s")
    os.makedirs(folder, exist_ok=True)

    pictures = {
        "1_undistorted.jpg": stages["undistorted"],
        "2_roi_and_warp_area.jpg": stages["roi"],
        "3_colour_mask.png": stages["colour"],
        "4_edge_mask.png": stages["edges"],
        "5_combined_mask_in_roi.png": stages["combined"],
        "6_birdseye_view.jpg": stages["warped"],
        "7_birdseye_mask.png": stages["binary_warped"],
        "8_lane_pixel_search.jpg": stages["search"],
        "9_final_overlay.jpg": stages["final"],
        "all_stages.jpg": diagnostic_grid(stages),
    }
    for filename, image in pictures.items():
        cv2.imwrite(os.path.join(folder, filename), image)
    print(f"Saved {len(pictures)} pictures to {folder}")


if __name__ == "__main__":
    main()
