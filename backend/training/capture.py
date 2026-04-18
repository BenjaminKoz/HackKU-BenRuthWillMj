"""Record ASL landmark samples from the webcam into data/landmarks.csv.

Controls while running (focus must be on the webcam window):
    a-z : record one sample labeled with that letter
    ESC : quit

This lets the team bootstrap a classifier in minutes without downloading a dataset.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import mediapipe as mp

OUT = Path(__file__).resolve().parents[2] / "data" / "landmarks.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils

    held_label = "A"
    f = OUT.open("a", newline="")
    writer = csv.writer(f)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            landmarks = None
            if res.multi_hand_landmarks:
                landmarks = res.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Last: {held_label}  (a-z = record, ESC = quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ASL capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if ord("a") <= key <= ord("z") and landmarks is not None:
                held_label = chr(key).upper()
                row = [held_label]
                for lm in landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)
                f.flush()
                print(f"recorded {held_label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        f.close()


if __name__ == "__main__":
    main()
