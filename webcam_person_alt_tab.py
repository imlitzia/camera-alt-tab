"""Detect a person with a webcam and press Alt+Tab on Windows.

Install:
    py -m pip install opencv-python

Run:
    py webcam_person_alt_tab.py

Choose a detected camera when prompted, then switch to your game during the
countdown. The program runs in the background without a preview by default.
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import time

import cv2


VK_MENU = 0x12  # Alt
VK_TAB = 0x09
KEYEVENTF_KEYUP = 0x0002


def press_alt_tab() -> None:
    """Send one Alt+Tab key combination through the Windows API."""
    if platform.system() != "Windows":
        raise RuntimeError("Alt+Tab automation in this program requires Windows.")

    send_key = ctypes.windll.user32.keybd_event
    send_key(VK_MENU, 0, 0, 0)
    send_key(VK_TAB, 0, 0, 0)
    send_key(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
    send_key(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Press Alt+Tab as soon as the webcam detects a person."
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
        "--startup-delay",
        type=float,
        default=5.0,
        help="Seconds to wait so you can return to your game (default: 5)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the camera preview (this window can take focus)",
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
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam {camera_index}.")

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if args.startup_delay > 0:
        print(f"\nSwitch to your game now. Detection starts in {args.startup_delay:g} seconds...")
        time.sleep(args.startup_delay)

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

            # A smaller frame lowers detection latency while retaining enough detail.
            scale = min(1.0, 640.0 / frame.shape[1])
            scan = cv2.resize(frame, None, fx=scale, fy=scale)
            boxes, _ = detector.detectMultiScale(
                scan,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            person_found = len(boxes) > 0
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
                cv2.putText(scan, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                for x, y, width, height in boxes:
                    cv2.rectangle(scan, (x, y), (x + width, y + height), color, 2)
                cv2.imshow("Person detector - Q to quit", scan)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()