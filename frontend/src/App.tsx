import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import {
  classify,
  classifyWordClip,
  compose,
  resetClassify,
  speak,
  type ClassifyMode,
} from "./lib/api";

const LETTER_CLASSIFY_INTERVAL_MS = 200;
const LETTER_POLL_INTERVAL_MS = 100;
const STABLE_FRAMES_TO_COMMIT = 6;
const MIN_CONFIDENCE = 0.7;

// Word capture must match training: capture.py records 90 frames at webcam rate (~30Hz).
const WORD_CLIP_FRAMES = 90;
const WORD_CAPTURE_INTERVAL_MS = 33;
const WORD_MIN_CONFIDENCE = 0.35;
const WORD_RESULT_PIN_MS = 1500;

type WordState = "idle" | "recording" | "classifying" | "result";

export default function App() {
  const [mode, setMode] = useState<ClassifyMode>("letters");
  const [currentLetter, setCurrentLetter] = useState<string>("-");
  const [confidence, setConfidence] = useState(0);
  const [buffer, setBuffer] = useState("");
  const [sentence, setSentence] = useState("");
  const [composing, setComposing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [status, setStatus] = useState("Waiting for hand…");

  // Word-mode push-to-record state.
  const [wordState, setWordState] = useState<WordState>("idle");
  const [recordProgress, setRecordProgress] = useState(0);
  const [lastWord, setLastWord] = useState<string | null>(null);
  const [lastWordConf, setLastWordConf] = useState(0);

  const modeRef = useRef<ClassifyMode>("letters");
  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const inflightRef = useRef(false);

  const recordedFramesRef = useRef<Landmark[][]>([]);
  const recordWatchdogRef = useRef<number | null>(null);
  const resultPinTimeoutRef = useRef<number | null>(null);
  const wordStateRef = useRef<WordState>("idle");
  const finishRecordingRef = useRef<() => void>(() => {});

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    wordStateRef.current = wordState;
  }, [wordState]);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
    // When recording, append every landmarker callback directly — that way the
    // clip's frame rate exactly matches the webcam rate the training data was
    // recorded at, without setInterval phase drift causing duplicates.
    if (wordStateRef.current !== "recording") return;
    const frames = recordedFramesRef.current;
    if (frames.length >= WORD_CLIP_FRAMES) return;
    if (lms) {
      frames.push(lms);
    } else if (frames.length > 0) {
      // Hand briefly lost — duplicate the last frame to keep timing intact
      // (matches capture.py:73-77).
      frames.push(frames[frames.length - 1]);
    } else {
      // No hand yet and no frames to duplicate; just wait for the next tick.
      return;
    }
    setRecordProgress(frames.length);
    if (frames.length >= WORD_CLIP_FRAMES) {
      finishRecordingRef.current();
    }
  }, []);

  const clearWordTimers = useCallback(() => {
    if (recordWatchdogRef.current !== null) {
      clearTimeout(recordWatchdogRef.current);
      recordWatchdogRef.current = null;
    }
    if (resultPinTimeoutRef.current !== null) {
      clearTimeout(resultPinTimeoutRef.current);
      resultPinTimeoutRef.current = null;
    }
  }, []);

  const switchMode = useCallback(
    (next: ClassifyMode) => {
      if (next === modeRef.current) return;
      modeRef.current = next;
      setMode(next);
      setBuffer("");
      setSentence("");
      setCurrentLetter("-");
      setConfidence(0);
      stableLetterRef.current = { letter: "", count: 0 };
      clearWordTimers();
      recordedFramesRef.current = [];
      wordStateRef.current = "idle";
      setWordState("idle");
      setRecordProgress(0);
      setLastWord(null);
      setLastWordConf(0);
      resetClassify().catch(() => {});
    },
    [clearWordTimers],
  );

  const finishRecording = useCallback(async () => {
    clearWordTimers();
    // Flip the ref synchronously so any landmarker callback that fires before
    // React commits setWordState won't start a fresh accumulation.
    wordStateRef.current = "classifying";
    const frames = recordedFramesRef.current;
    recordedFramesRef.current = [];
    if (frames.length !== WORD_CLIP_FRAMES) {
      setStatus(`Recording too short (${frames.length}/${WORD_CLIP_FRAMES}) — try again.`);
      wordStateRef.current = "idle";
      setWordState("idle");
      setRecordProgress(0);
      return;
    }
    setWordState("classifying");
    setStatus("Classifying…");
    try {
      const { letter, confidence: conf } = await classifyWordClip(frames);
      setLastWord(letter);
      setLastWordConf(conf);
      wordStateRef.current = "result";
      setWordState("result");
      if (conf >= WORD_MIN_CONFIDENCE) {
        setBuffer((b) => (b ? `${b} ${letter}` : letter));
        setStatus(`Recognized "${letter}" (${(conf * 100).toFixed(0)}%)`);
      } else {
        setStatus(`Low confidence: "${letter}" (${(conf * 100).toFixed(0)}%) — not added`);
      }
      resultPinTimeoutRef.current = window.setTimeout(() => {
        wordStateRef.current = "idle";
        setWordState("idle");
        setRecordProgress(0);
        resultPinTimeoutRef.current = null;
      }, WORD_RESULT_PIN_MS);
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "classify error");
      wordStateRef.current = "idle";
      setWordState("idle");
      setRecordProgress(0);
    }
  }, [clearWordTimers]);

  // Keep finishRecordingRef in sync so the landmarker callback can call the
  // latest version without being re-bound on every render.
  useEffect(() => {
    finishRecordingRef.current = finishRecording;
  }, [finishRecording]);

  const startRecording = useCallback(() => {
    if (modeRef.current !== "words") return;
    if (wordStateRef.current !== "idle") return;
    if (!lastLandmarksRef.current) {
      setStatus("No hand detected — show your hand before recording.");
      return;
    }
    clearWordTimers();
    recordedFramesRef.current = [];
    setLastWord(null);
    setLastWordConf(0);
    setRecordProgress(0);
    wordStateRef.current = "recording";
    setWordState("recording");
    setStatus("Recording… sign now");

    // Safety net: if for some reason the landmarker stalls (e.g. hand leaves
    // before any frame was captured), give up after a generous timeout so the
    // UI doesn't get stuck in "recording" forever.
    const maxMs = WORD_CAPTURE_INTERVAL_MS * WORD_CLIP_FRAMES * 2;
    recordWatchdogRef.current = window.setTimeout(() => {
      if (wordStateRef.current === "recording") {
        setStatus(
          `Recording timed out (${recordedFramesRef.current.length}/${WORD_CLIP_FRAMES} frames) — try again.`,
        );
        recordedFramesRef.current = [];
        wordStateRef.current = "idle";
        setWordState("idle");
        setRecordProgress(0);
      }
    }, maxMs);
  }, [clearWordTimers]);

  // Space bar in Words mode triggers a recording (unless typing in a field).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      if (modeRef.current !== "words") return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      e.preventDefault();
      startRecording();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [startRecording]);

  useEffect(() => {
    return () => clearWordTimers();
  }, [clearWordTimers]);

  // Continuous per-frame classify, letter mode only.
  useEffect(() => {
    const id = setInterval(async () => {
      if (modeRef.current !== "letters") return;
      const lms = lastLandmarksRef.current;
      const now = performance.now();
      if (!lms || inflightRef.current || now - lastClassifyRef.current < LETTER_CLASSIFY_INTERVAL_MS) {
        return;
      }
      lastClassifyRef.current = now;
      inflightRef.current = true;
      try {
        const { letter, confidence: conf } = await classify(lms, "letters");
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
    }, LETTER_POLL_INTERVAL_MS);
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
      const text = await compose(buffer, mode);
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

  const bufferPlaceholder = mode === "words" ? "sign a word…" : "start signing…";
  const bufferLabel = mode === "words" ? "Word buffer" : "Letter buffer";
  const recordPct = (recordProgress / WORD_CLIP_FRAMES) * 100;

  const wordBigLabel =
    wordState === "recording"
      ? "●REC"
      : wordState === "classifying"
      ? "…"
      : wordState === "result" && lastWord
      ? lastWord
      : "—";

  return (
    <div className="app">
      <div className="header">
        <h1>ASL Translator</h1>
        <span className="tag">HackKU 2026 · Ben · Ruth · Will · MJ</span>
      </div>

      <div className="row" style={{ justifyContent: "center", marginBottom: 16 }}>
        <button
          onClick={() => switchMode("letters")}
          className={mode === "letters" ? "" : "secondary"}
        >
          Letters
        </button>
        <button
          onClick={() => switchMode("words")}
          className={mode === "words" ? "" : "secondary"}
        >
          Words
        </button>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Camera</h2>
          <Webcam onLandmarks={handleLandmarks} />

          {mode === "letters" ? (
            <>
              <div className="letter-big">{currentLetter}</div>
              <div className="confidence">confidence {(confidence * 100).toFixed(0)}%</div>
            </>
          ) : (
            <>
              <div
                className="letter-big"
                style={{
                  color:
                    wordState === "result" && lastWordConf >= WORD_MIN_CONFIDENCE
                      ? "var(--accent-2)"
                      : wordState === "recording"
                      ? "#ef4444"
                      : undefined,
                }}
              >
                {wordBigLabel}
              </div>
              {wordState === "result" && (
                <div className="confidence">
                  confidence {(lastWordConf * 100).toFixed(0)}%
                </div>
              )}
              <div
                style={{
                  height: 10,
                  borderRadius: 6,
                  background: "#1e293b",
                  overflow: "hidden",
                  marginTop: 8,
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${recordPct}%`,
                    background: wordState === "recording" ? "#ef4444" : "#334155",
                    transition: "width 50ms linear",
                  }}
                />
              </div>
              <div style={{ marginTop: 12, display: "flex", justifyContent: "center" }}>
                <button
                  onClick={startRecording}
                  disabled={wordState !== "idle"}
                  style={{ padding: "10px 20px", fontSize: 16 }}
                >
                  {wordState === "idle"
                    ? "🎙 Record word (Space)"
                    : wordState === "recording"
                    ? `Recording… ${recordProgress}/${WORD_CLIP_FRAMES}`
                    : wordState === "classifying"
                    ? "Classifying…"
                    : "Done"}
                </button>
              </div>
            </>
          )}
          <div className="status">{status}</div>
        </div>

        <div className="panel">
          <h2>{bufferLabel}</h2>
          <div className="buffer">
            {buffer || <span style={{ color: "#8892a6" }}>{bufferPlaceholder}</span>}
          </div>
          <div className="row">
            <button className="secondary" onClick={onSpace}>Space</button>
            <button className="secondary" onClick={onBackspace}>Backspace</button>
            <button className="secondary" onClick={onClear}>Clear</button>
            <button onClick={onCompose} disabled={composing || !buffer.trim()}>
              {composing ? "Composing…" : "Compose sentence"}
            </button>
          </div>

          <h2 style={{ marginTop: 24 }}>Sentence</h2>
          <div className="sentence">
            {sentence || (
              <span style={{ color: "#8892a6" }}>
                Press "Compose sentence" to turn {mode === "words" ? "words" : "letters"} into English with Gemini.
              </span>
            )}
          </div>
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
