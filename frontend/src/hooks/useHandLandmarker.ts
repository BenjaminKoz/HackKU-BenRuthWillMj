import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

export type Landmark = { x: number; y: number; z: number };

export type HandLandmarkerCallbacks = {
  // Single-hand callback for existing consumers (letter mode, Learning, Game).
  // Receives the leftmost detected hand, or null if none.
  onLandmarks?: (lms: Landmark[] | null) => void;
  // Multi-hand callback: receives all detected hands (0, 1, or 2), sorted
  // leftmost-wrist-x first. Used by word mode.
  onHands?: (hands: Landmark[][]) => void;
};

export function useHandLandmarker(
  videoRef: React.RefObject<HTMLVideoElement>,
  callbacks: HandLandmarkerCallbacks,
) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);
  // Stash the latest callbacks in a ref so the detection loop doesn't rebind
  // every time the parent re-renders.
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
        );
        const lm = await HandLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
        });
        if (cancelled) {
          lm.close();
          return;
        }
        landmarkerRef.current = lm;
        setReady(true);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    const video = videoRef.current;
    if (!video) return;

    const tick = () => {
      const lm = landmarkerRef.current;
      if (!lm || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const now = performance.now();
      if (now - lastTimeRef.current < 33) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      lastTimeRef.current = now;
      const res: HandLandmarkerResult = lm.detectForVideo(video, now);

      const rawHands = (res.landmarks ?? []) as Landmark[][];
      // Canonical order: leftmost wrist (landmark 0) first. Matches the
      // sort_hands_by_x invariant the word model is trained against.
      const sorted = rawHands
        .slice()
        .sort((a, b) => a[0].x - b[0].x);

      const { onLandmarks, onHands } = callbacksRef.current;
      onHands?.(sorted);
      onLandmarks?.(sorted.length > 0 ? sorted[0] : null);

      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [ready, videoRef]);

  return { ready, error };
}
