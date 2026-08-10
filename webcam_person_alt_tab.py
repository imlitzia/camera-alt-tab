"""Detect a person with a webcam and press Alt+Tab on Windows.

Install:
    py -m pip install opencv-python

Run:
    py webcam_person_alt_tab.py

Press Q in the preview window to stop. Use --no-preview to run without it.
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
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
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
    parser.add_argument("--no-preview", action="store_true", help="Hide webcam preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if platform.system() != "Windows":
        raise SystemExit("This program currently supports Windows only.")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}.")

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    armed = True
    last_trigger = float("-inf")
    absent_since: float | None = None

    print("Watching for a person. Press Q in the preview window or Ctrl+C to stop.")
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

            if not args.no_preview:
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
