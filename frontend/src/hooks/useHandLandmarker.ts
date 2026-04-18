import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

export type Landmark = { x: number; y: number; z: number };

export function useHandLandmarker(
  videoRef: React.RefObject<HTMLVideoElement>,
  onLandmarks: (lms: Landmark[] | null) => void
) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);

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
          numHands: 1,
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
      if (res.landmarks && res.landmarks.length > 0) {
        onLandmarks(res.landmarks[0] as Landmark[]);
      } else {
        onLandmarks(null);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [ready, videoRef, onLandmarks]);

  return { ready, error };
}
