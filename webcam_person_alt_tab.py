"""Detect a person and send a normal Alt+Tab on Windows.

Install:
    py -m pip install opencv-python ultralytics

Run:
    py webcam_person_alt_tab.py

Choose a detected camera when prompted, then activate your game or any other
application. Detection sends Alt+Tab to whichever application is active.
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import time

import cv2
from ultralytics import YOLO


VK_MENU = 0x12
VK_TAB = 0x09
KEYEVENTF_KEYUP = 0x0002


def press_alt_tab() -> None:
    """Send one normal Alt+Tab to the currently active application."""
    send_key = ctypes.windll.user32.keybd_event
    send_key(VK_MENU, 0, 0, 0)
    send_key(VK_TAB, 0, 0, 0)
    send_key(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
    send_key(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Alt+Tab when the webcam detects a person."
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Skip the menu and use this webcam index",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between Alt+Tab actions (default: 2)",
    )
    parser.add_argument(
        "--reset-after",
        type=float,
        default=1.0,
        help="Person must be absent this long before another trigger (default: 1)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the camera preview (this window can take focus)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO person confidence from 0 to 1 (default: 0.35)",
    )
    return parser.parse_args()


def find_available_cameras(max_index: int = 10) -> list[int]:
    """Return camera indexes that can provide a frame."""
    available: list[int] = []
    print("Scanning for cameras...")
    for index in range(max_index):
        test_camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if test_camera.isOpened():
            ok, _ = test_camera.read()
            if ok:
                available.append(index)
        test_camera.release()
    return available


def choose_camera() -> int:
    """Scan for cameras and ask the user to choose one."""
    cameras = find_available_cameras()
    if not cameras:
        raise SystemExit("No working cameras were detected.")

    print("\nAvailable cameras:")
    for index in cameras:
        print(f"  [{index}] Camera {index}")

    while True:
        answer = input("\nEnter the camera number to use: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            print("Please enter one of the camera numbers shown above.")
            continue
        if selected in cameras:
            return selected
        print("That camera is unavailable. Choose a number from the list.")


def main() -> None:
    args = parse_args()
    if platform.system() != "Windows":
        raise SystemExit("This program currently supports Windows only.")

    camera_index = args.camera if args.camera is not None else choose_camera()
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam {camera_index}.")

    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence must be greater than 0 and no more than 1.")

    print("Loading YOLO person detector...")
    detector = YOLO("yolov8n.pt")

    armed = True
    last_trigger = float("-inf")
    absent_since: float | None = None

    if args.preview:
        print("Watching for a person. Press Q in the preview window or Ctrl+C to stop.")
    else:
        print("Watching for a person in the background. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                time.sleep(0.02)
                continue

            # COCO class 0 is "person". The nano model is optimized for low-latency
            # webcam inference and is much more reliable than Haar/HOG detection.
            result = detector.predict(
                source=frame,
                classes=[0],
                conf=args.confidence,
                imgsz=640,
                verbose=False,
            )[0]
            person_boxes = result.boxes
            person_found = len(person_boxes) > 0
            now = time.monotonic()

            if person_found:
                absent_since = None
                if armed and now - last_trigger >= args.cooldown:
                    press_alt_tab()
                    last_trigger = now
                    armed = False
                    print("Person detected: Alt+Tab pressed.")
            else:
                if absent_since is None:
                    absent_since = now
                elif not armed and now - absent_since >= args.reset_after:
                    armed = True

            if args.preview:
                color = (0, 255, 0) if person_found else (0, 165, 255)
                label = "PERSON DETECTED" if person_found else "Watching..."
                cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if person_found:
                    for x1, y1, x2, y2 in person_boxes.xyxy.cpu().numpy().astype(int):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.imshow("Person detector - Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()