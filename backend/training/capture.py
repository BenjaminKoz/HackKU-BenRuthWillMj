"""Record ASL landmark samples from the webcam.

Static single-frame samples -> data/landmarks.csv          (one hand)
Word sequences              -> data/landmarks_words.csv    (TWO hands per frame:
                                                            leftmost-wrist first,
                                                            missing slot zero-padded)

Controls while running (focus must be on the webcam window):
    a-z              record one static sample labeled with that letter
    TAB              toggle word-capture mode. In word mode:
        Left/Right arrow  cycle through the target word list
        SPACE             record a WORD_FRAMES clip labeled with the selected word
        TAB               back to letter mode
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
WORDS_CSV = DATA_DIR / "landmarks_words.csv"

WORD_FRAMES = 90    # ~3 seconds at 30fps — room for slower two-movement signs

# Words still to record. Already-trained words (HELLO, THANKS, YES, NO, PLEASE,
# SORRY, LOVE, YOU, HELP, NAME) are omitted so the arrows don't cycle past them.
# Ordered by demo priority:
#   Tier 1 (unlock core sentences): I, WANT, NEED, GOOD, MORE
#   Tier 2 (expressiveness):        FRIEND, UNDERSTAND, HOW, EAT, HAPPY
#   Tier 3 (polish):                WHY, WHERE, DRINK, FAMILY, HOME
# If you need to top up samples for an already-trained word, add it back here.
WORDS = [
    "I", "WANT", "NEED", "GOOD", "MORE",
    "FRIEND", "UNDERSTAND", "HOW", "EAT", "HAPPY",
    "WHY", "WHERE", "DRINK", "FAMILY", "HOME",
]


def flatten_landmarks(lms) -> list[float]:
    out = []
    for lm in lms.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out


def hand_to_array(lms) -> list[list[float]]:
    """Return a single hand as a list of 21 [x, y, z] triples."""
    return [[lm.x, lm.y, lm.z] for lm in lms.landmark]


def two_hand_frame_floats(multi_hand_landmarks) -> list[float]:
    """Encode up to 2 MediaPipe hands for a single frame into the fixed 2-slot
    layout used by train_words.py: leftmost-wrist-x first, missing slot = zeros.
    Returns 2 * 21 * 3 = 126 floats."""
    hands = [hand_to_array(h) for h in (multi_hand_landmarks or [])][:2]
    # Sort by wrist (landmark 0) x, leftmost first — same canonical order as
    # app/services/features.py sort_hands_by_x.
    hands.sort(key=lambda h: h[0][0])
    while len(hands) < 2:
        hands.append([[0.0, 0.0, 0.0] for _ in range(21)])
    out: list[float] = []
    for hand in hands:
        for lm in hand:
            out.extend(lm)
    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: couldn't open webcam (cv2.VideoCapture(0) failed).")
        print("Likely cause: another process is using it — usually the frontend")
        print("browser tab. Close http://localhost:5173 (or stop `npm run dev`)")
        print("and try again.")
        return

    # max_num_hands=2 so word mode can record two-handed signs. Single-hand
    # letter/motion capture still just takes multi_hand_landmarks[0].
    hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
    draw = mp.solutions.drawing_utils

    held_label = "A"
    static_f = STATIC_CSV.open("a", newline="")
    static_writer = csv.writer(static_f)
    words_f = WORDS_CSV.open("a", newline="")
    words_writer = csv.writer(words_f)

    word_recording: dict | None = None  # {'label': 'HELLO', 'frames': [[126 floats], ...], 'target': WORD_FRAMES}
    word_mode = False
    word_index = 0  # index into WORDS

    # cv2.waitKeyEx returns a platform-specific extended code for arrow keys.
    # Windows uses 24-bit codes; Linux/X11 uses 655xx. Support both.
    LEFT_ARROW_CODES = {2424832, 65361, 0x51}
    RIGHT_ARROW_CODES = {2555904, 65363, 0x53}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: cap.read() returned no frame — camera may have been claimed by another app.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            landmarks = None
            multi_hands = res.multi_hand_landmarks or []
            if multi_hands:
                landmarks = multi_hands[0]
                for h in multi_hands:
                    draw.draw_landmarks(frame, h, mp.solutions.hands.HAND_CONNECTIONS)

            # If in the middle of a word capture, grab the next frame's two-hand
            # landmarks (sorted leftmost-first, zero-padded if only one is visible).
            # A lost hand mid-clip duplicates the last frame so timing stays intact.
            if word_recording is not None:
                if multi_hands:
                    word_recording["frames"].append(two_hand_frame_floats(multi_hands))
                elif word_recording["frames"]:
                    word_recording["frames"].append(word_recording["frames"][-1])
                # else: hand not yet visible; wait one more loop.

                if len(word_recording["frames"]) >= word_recording["target"]:
                    row = [word_recording["label"]]
                    for flat in word_recording["frames"]:
                        row.extend(flat)
                    words_writer.writerow(row)
                    words_f.flush()
                    print(f"recorded word {word_recording['label']}")
                    word_recording = None

            if word_recording is not None:
                progress = len(word_recording["frames"])
                cv2.putText(
                    frame,
                    f"REC {word_recording['label']}  {progress}/{word_recording['target']}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
            elif word_mode:
                target_word = WORDS[word_index]
                cv2.putText(
                    frame,
                    f"WORD MODE  target: {target_word}  ({word_index + 1}/{len(WORDS)})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Left/Right arrow select, SPACE record, TAB back to letters",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                )
            else:
                cv2.putText(
                    frame,
                    f"Last: {held_label}  (a-z static, TAB words, ESC quit)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("ASL capture", frame)

            # waitKeyEx returns full extended scancodes (needed for arrow keys).
            # For ASCII-range keys we still want a 0-255 value, so mask.
            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF if key_ex != -1 else 255
            if key == 27:  # ESC
                break

            # TAB toggles word mode. Disabled mid-clip.
            if word_recording is None and key == 9:  # TAB
                word_mode = not word_mode
                print(f"word mode: {'ON' if word_mode else 'OFF'}")
                continue

            if word_mode and word_recording is None:
                # Arrow keys cycle through WORDS (wraps around at the ends).
                if key_ex in LEFT_ARROW_CODES:
                    word_index = (word_index - 1) % len(WORDS)
                    print(f"selected word: {WORDS[word_index]}")
                    continue
                if key_ex in RIGHT_ARROW_CODES:
                    word_index = (word_index + 1) % len(WORDS)
                    print(f"selected word: {WORDS[word_index]}")
                    continue
                # SPACE starts a word-clip recording.
                if key == ord(" "):
                    if landmarks is None:
                        print(f"can't start word {WORDS[word_index]}: no hand detected")
                    else:
                        word_recording = {
                            "label": WORDS[word_index],
                            "frames": [],
                            "target": WORD_FRAMES,
                        }
                        print(f"starting word capture for {WORDS[word_index]}; sign it now")
                    continue
                # In word mode, letter keys are ignored — force TAB to exit first.
                continue

            # Static capture: lowercase a-z. Disabled during word recording.
            if (
                word_recording is None
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
        words_f.close()


if __name__ == "__main__":
    main()
