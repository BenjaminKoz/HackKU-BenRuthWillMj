"""Record ASL landmark samples from the webcam.

Static single-frame samples -> data/landmarks.csv
Motion sequences (J, Z)     -> data/landmarks_motion.csv

Controls while running (focus must be on the webcam window):
    a-z              record one static sample labeled with that letter
    Shift+J, Shift+Z record a 70-frame motion clip for J or Z
                     (start signing the motion the instant you press the key;
                     keep your hand in frame until the counter hits 70 — the
                     training pipeline can shorten longer clips but can't extend
                     shorter ones, so we err on the side of too many frames)
    ESC              quit
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import mediapipe as mp

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CSV = DATA_DIR / "landmarks.csv"
MOTION_CSV = DATA_DIR / "landmarks_motion.csv"

MOTION_FRAMES = 70  # ~2.3 seconds at 30fps — headroom for Z and most short words


def flatten_landmarks(lms) -> list[float]:
    out = []
    for lm in lms.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: couldn't open webcam (cv2.VideoCapture(0) failed).")
        print("Likely cause: another process is using it — usually the frontend")
        print("browser tab. Close http://localhost:5173 (or stop `npm run dev`)")
        print("and try again.")
        return

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils

    held_label = "A"
    static_f = STATIC_CSV.open("a", newline="")
    static_writer = csv.writer(static_f)
    motion_f = MOTION_CSV.open("a", newline="")
    motion_writer = csv.writer(motion_f)

    motion_recording: dict | None = None  # {'label': 'J', 'frames': [[63 floats], ...]}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: cap.read() returned no frame — camera may have been claimed by another app.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            landmarks = None
            if res.multi_hand_landmarks:
                landmarks = res.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # If in the middle of a motion capture, grab the next frame's landmarks.
            # A lost hand mid-clip duplicates the last frame so timing stays intact.
            if motion_recording is not None:
                if landmarks is not None:
                    motion_recording["frames"].append(flatten_landmarks(landmarks))
                elif motion_recording["frames"]:
                    motion_recording["frames"].append(motion_recording["frames"][-1])
                # else: hand not yet visible; wait one more loop.

                if len(motion_recording["frames"]) >= MOTION_FRAMES:
                    row = [motion_recording["label"]]
                    for flat in motion_recording["frames"]:
                        row.extend(flat)
                    motion_writer.writerow(row)
                    motion_f.flush()
                    print(f"recorded motion {motion_recording['label']}")
                    motion_recording = None

            if motion_recording is not None:
                progress = len(motion_recording["frames"])
                cv2.putText(
                    frame,
                    f"REC {motion_recording['label']}  {progress}/{MOTION_FRAMES}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    f"Last: {held_label}  (a-z static, Shift+J/Z motion, ESC quit)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("ASL capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Motion trigger: Shift+J (74) or Shift+Z (90). Requires hand already
            # in frame so the clip starts with a valid pose.
            if motion_recording is None and key in (ord("J"), ord("Z")):
                if landmarks is None:
                    print(f"can't start motion {chr(key)}: no hand detected")
                else:
                    motion_recording = {"label": chr(key), "frames": []}
                    print(f"starting motion capture for {chr(key)}; sign it now")
                continue

            # Static capture: lowercase a-z. Disabled during motion recording.
            if (
                motion_recording is None
                and ord("a") <= key <= ord("z")
                and landmarks is not None
            ):
                held_label = chr(key).upper()
                row = [held_label] + flatten_landmarks(landmarks)
                static_writer.writerow(row)
                static_f.flush()
                print(f"recorded {held_label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        static_f.close()
        motion_f.close()


if __name__ == "__main__":
    main()
