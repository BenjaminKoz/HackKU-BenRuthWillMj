import { useEffect, useRef } from "react";
import { useHandLandmarker, type Landmark } from "../hooks/useHandLandmarker";

type Props = {
  onLandmarks: (lms: Landmark[] | null) => void;
};

const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export function Webcam({ onLandmarks }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastLmsRef = useRef<Landmark[] | null>(null);

  const handleLms = (lms: Landmark[] | null) => {
    lastLmsRef.current = lms;
    onLandmarks(lms);
    drawOverlay();
  };

  const { ready, error } = useHandLandmarker(videoRef, handleLms);

  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          resizeCanvas();
        }
      } catch (e) {
        console.error("getUserMedia failed", e);
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const resizeCanvas = () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    c.width = v.videoWidth || 640;
    c.height = v.videoHeight || 480;
  };

  const drawOverlay = () => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    const lms = lastLmsRef.current;
    if (!lms) return;

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 3;
    for (const [a, b] of HAND_EDGES) {
      ctx.beginPath();
      ctx.moveTo(lms[a].x * c.width, lms[a].y * c.height);
      ctx.lineTo(lms[b].x * c.width, lms[b].y * c.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#7c5cff";
    for (const lm of lms) {
      ctx.beginPath();
      ctx.arc(lm.x * c.width, lm.y * c.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  return (
    <div className="video-wrap">
      <video ref={videoRef} playsInline muted onLoadedMetadata={resizeCanvas} />
      <canvas ref={canvasRef} />
      {!ready && !error && (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#8892a6" }}>
          Loading hand tracker…
        </div>
      )}
      {error && (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#ff7070", padding: 16, textAlign: "center" }}>
          {error}
        </div>
      )}
    </div>
  );
}
