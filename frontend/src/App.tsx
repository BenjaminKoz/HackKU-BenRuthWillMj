import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify, compose, speak } from "./lib/api";

const CLASSIFY_INTERVAL_MS = 200;
const STABLE_FRAMES_TO_COMMIT = 6;
const MIN_CONFIDENCE = 0.7;

export default function App() {
  const [currentLetter, setCurrentLetter] = useState<string>("-");
  const [confidence, setConfidence] = useState(0);
  const [buffer, setBuffer] = useState("");
  const [sentence, setSentence] = useState("");
  const [composing, setComposing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [status, setStatus] = useState("Waiting for hand…");

  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const inflightRef = useRef(false);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      const lms = lastLandmarksRef.current;
      const now = performance.now();
      if (!lms || inflightRef.current || now - lastClassifyRef.current < CLASSIFY_INTERVAL_MS) return;
      lastClassifyRef.current = now;
      inflightRef.current = true;
      try {
        const { letter, confidence: conf } = await classify(lms);
        setCurrentLetter(letter);
        setConfidence(conf);
        if (conf >= MIN_CONFIDENCE) {
          const prev = stableLetterRef.current;
          if (prev.letter === letter) {
            prev.count += 1;
          } else {
            stableLetterRef.current = { letter, count: 1 };
          }
          if (
            stableLetterRef.current.count === STABLE_FRAMES_TO_COMMIT &&
            letter !== "nothing"
          ) {
            setBuffer((b) => (b.endsWith(letter) ? b : b + letter));
          }
        }
        setStatus("Tracking");
      } catch (e) {
        setStatus(e instanceof Error ? e.message : "classify error");
      } finally {
        inflightRef.current = false;
      }
    }, 100);
    return () => clearInterval(id);
  }, []);

  const onSpace = () => setBuffer((b) => b + " ");
  const onBackspace = () => setBuffer((b) => b.slice(0, -1));
  const onClear = () => {
    setBuffer("");
    setSentence("");
  };

  const onCompose = async () => {
    if (!buffer.trim()) return;
    setComposing(true);
    try {
      const text = await compose(buffer);
      setSentence(text);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "compose error");
    } finally {
      setComposing(false);
    }
  };

  const onSpeak = async () => {
    if (!sentence.trim()) return;
    setSpeaking(true);
    try {
      const blob = await speak(sentence);
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.onended = () => URL.revokeObjectURL(url);
      await audio.play();
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "speak error");
    } finally {
      setSpeaking(false);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <h1>ASL Translator</h1>
        <span className="tag">HackKU 2026 · Ben · Ruth · Will · MJ</span>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Camera</h2>
          <Webcam onLandmarks={handleLandmarks} />
          <div className="letter-big">{currentLetter}</div>
          <div className="confidence">confidence {(confidence * 100).toFixed(0)}%</div>
          <div className="status">{status}</div>
        </div>

        <div className="panel">
          <h2>Letter buffer</h2>
          <div className="buffer">{buffer || <span style={{ color: "#8892a6" }}>start signing…</span>}</div>
          <div className="row">
            <button className="secondary" onClick={onSpace}>Space</button>
            <button className="secondary" onClick={onBackspace}>Backspace</button>
            <button className="secondary" onClick={onClear}>Clear</button>
            <button onClick={onCompose} disabled={composing || !buffer.trim()}>
              {composing ? "Composing…" : "Compose sentence"}
            </button>
          </div>

          <h2 style={{ marginTop: 24 }}>Sentence</h2>
          <div className="sentence">{sentence || <span style={{ color: "#8892a6" }}>Press "Compose sentence" to turn letters into English with Gemini.</span>}</div>
          <div className="row">
            <button onClick={onSpeak} disabled={speaking || !sentence.trim()}>
              {speaking ? "Speaking…" : "🔊 Speak (ElevenLabs)"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
