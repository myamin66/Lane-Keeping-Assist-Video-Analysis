"""
run_lka.py

Main script. Reads a road video, runs the lane keeping assist perception
on every frame and writes the annotated video.

Examples:
    python run_lka.py videos/project_video.mp4
    python run_lka.py videos/challenge_video.mp4 --camera challenge --diagnostic
    python run_lka.py videos/project_video.mp4 --start 20 --end 30 --output output/bridge.mp4

By default the result goes to output/<video name>_lka.mp4 and, with
--diagnostic, also output/<video name>_diagnostic.mp4.
"""

import argparse
import os
import time

import cv2

from pipeline import LaneKeepingAssist, diagnostic_grid


def parse_arguments():
    parser = argparse.ArgumentParser(description="Lane keeping assist video analysis")
    parser.add_argument("video", help="input video file")
    parser.add_argument("--output", help="annotated output video (default: output/<name>_lka.mp4)")
    parser.add_argument("--diagnostic", action="store_true",
                        help="also write a video showing all pipeline stages")
    parser.add_argument("--camera", default="default", choices=["default", "challenge"],
                        help="camera mounting profile from settings.py "
                             "(use 'challenge' for challenge_video.mp4)")
    parser.add_argument("--start", type=float, default=0, help="start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="end time in seconds")
    return parser.parse_args()


def main():
    args = parse_arguments()

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    name = os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs("output", exist_ok=True)
    output_path = args.output or os.path.join("output", f"{name}_lka.mp4")
    diagnostic_path = os.path.splitext(output_path)[0].replace("_lka", "") + "_diagnostic.mp4"

    # mp4v plays in VLC and the Windows player. See README for converting
    # to H.264 if a browser or PowerPoint refuses to play it.
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, codec, fps, (1280, 720))
    diagnostic_writer = None
    if args.diagnostic:
        diagnostic_writer = cv2.VideoWriter(diagnostic_path, codec, fps, (1920, 720))

    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps) if args.end is not None else total_frames
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    lka = LaneKeepingAssist(args.camera)
    # Count how often each state shows up, printed at the end as a summary.
    counts = {"left": {}, "right": {}}

    started = time.time()
    frame_number = start_frame
    while frame_number < end_frame:
        ok, frame = video.read()
        if not ok:
            break

        annotated, stages = lka.process(frame)
        writer.write(annotated)
        if diagnostic_writer is not None:
            diagnostic_writer.write(diagnostic_grid(stages))

        for side in ("left", "right"):
            status = getattr(lka.tracker, side).status
            counts[side][status] = counts[side].get(status, 0) + 1

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"  frame {frame_number}/{end_frame}")

    video.release()
    writer.release()
    if diagnostic_writer is not None:
        diagnostic_writer.release()

    done = frame_number - start_frame
    seconds = time.time() - started
    print(f"Processed {done} frames in {seconds:.1f} s ({done / max(seconds, 1e-6):.1f} fps)")
    print(f"Saved {output_path}")
    if diagnostic_writer is not None:
        print(f"Saved {diagnostic_path}")
    for side in ("left", "right"):
        summary = ", ".join(f"{k} {100 * v / max(done, 1):.0f}%" for k, v in counts[side].items())
        print(f"  {side:5s}: {summary}")


if __name__ == "__main__":
    main()
