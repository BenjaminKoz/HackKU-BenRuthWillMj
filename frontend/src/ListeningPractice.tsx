import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify, speak } from "./lib/api";

const LETTERS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
  "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
  "V", "W", "X", "Y",
];

const CLASSIFY_INTERVAL_MS = 200;
const POLL_INTERVAL_MS = 100;
const STABLE_FRAMES_TO_COMMIT = 6;
const MIN_CONFIDENCE = 0.65;
const ADVANCE_DELAY_MS = 1200;

type RoundState = "idle" | "playing" | "listening" | "correct";

const PHRASINGS = [
  (l: string) => `The letter ${l}.`,
  (l: string) => `Please sign ${l}.`,
  (l: string) => `Can you show me ${l}?`,
  (l: string) => `Next letter: ${l}.`,
  (l: string) => `${l}, please.`,
];

function pickPhrase(letter: string) {
  const fn = PHRASINGS[Math.floor(Math.random() * PHRASINGS.length)];
  return fn(letter);
}

function pickLetter(prev: string | null) {
  let next = prev;
  while (next === prev) {
    next = LETTERS[Math.floor(Math.random() * LETTERS.length)];
  }
  return next!;
}

export function ListeningPractice() {
  const [target, setTarget] = useState<string | null>(null);
  const [phrase, setPhrase] = useState<string>("");
  const [roundState, setRoundState] = useState<RoundState>("idle");
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [bestStreak, setBestStreak] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [revealed, setRevealed] = useState(false);

  const [detected, setDetected] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Press Start to begin");

  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const inflightRef = useRef(false);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const advanceTimeoutRef = useRef<number | null>(null);

  const targetRef = useRef<string | null>(null);
  const roundStateRef = useRef<RoundState>("idle");

  useEffect(() => {
    targetRef.current = target;
  }, [target]);

  useEffect(() => {
    roundStateRef.current = roundState;
  }, [roundState]);

  const stopAudio = useCallback(() => {
    const a = audioRef.current;
    if (a) {
      a.pause();
      a.src = "";
      audioRef.current = null;
    }
  }, []);

  const playLetter = useCallback(async (letter: string) => {
    const text = pickPhrase(letter);
    setPhrase(text);
    setRoundState("playing");
    setStatus("🔊 Listening to client…");
    try {
      const blob = await speak(text);
      stopAudio();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        URL.revokeObjectURL(url);
        if (roundStateRef.current === "playing" && targetRef.current === letter) {
          setRoundState("listening");
          setStatus("Your turn — sign the letter you heard");
        }
      };
      await audio.play();
    } catch (e) {
      setStatus(e instanceof Error ? `Audio error: ${e.message}` : "Audio error");
      setRoundState("listening");
    }
  }, [stopAudio]);

  const nextRound = useCallback(() => {
    if (advanceTimeoutRef.current !== null) {
      clearTimeout(advanceTimeoutRef.current);
      advanceTimeoutRef.current = null;
    }
    setRevealed(false);
    stableLetterRef.current = { letter: "", count: 0 };
    const next = pickLetter(targetRef.current);
    setTarget(next);
    targetRef.current = next;
    void playLetter(next);
  }, [playLetter]);

  const replay = useCallback(() => {
    if (!targetRef.current) return;
    void playLetter(targetRef.current);
  }, [playLetter]);

  const skip = useCallback(() => {
    if (!targetRef.current) return;
    setStreak(0);
    setRevealed(true);
    setStatus(`Skipped — answer was ${targetRef.current}`);
    if (advanceTimeoutRef.current !== null) clearTimeout(advanceTimeoutRef.current);
    advanceTimeoutRef.current = window.setTimeout(() => {
      nextRound();
    }, ADVANCE_DELAY_MS);
  }, [nextRound]);

  const reveal = useCallback(() => {
    if (!targetRef.current) return;
    setStreak(0);
    setRevealed(true);
    setStatus(`The letter was ${targetRef.current}`);
  }, []);

  const reset = useCallback(() => {
    stopAudio();
    if (advanceTimeoutRef.current !== null) clearTimeout(advanceTimeoutRef.current);
    setScore(0);
    setStreak(0);
    setAttempts(0);
    setBestStreak(0);
    setRevealed(false);
    setTarget(null);
    targetRef.current = null;
    setPhrase("");
    setRoundState("idle");
    setStatus("Press Start to begin");
    stableLetterRef.current = { letter: "", count: 0 };
  }, [stopAudio]);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      if (roundStateRef.current !== "listening") return;
      const lms = lastLandmarksRef.current;
      const now = performance.now();
      if (!lms || inflightRef.current || now - lastClassifyRef.current < CLASSIFY_INTERVAL_MS) {
        return;
      }
      lastClassifyRef.current = now;
      inflightRef.current = true;
      try {
        const { letter, confidence: conf } = await classify(lms, "letters");
        const upper = letter.toUpperCase();
        setDetected(upper);
        setConfidence(conf);

        if (upper === "NOTHING" || conf < MIN_CONFIDENCE) {
          stableLetterRef.current = { letter: "", count: 0 };
          return;
        }

        const prev = stableLetterRef.current;
        if (prev.letter === upper) {
          prev.count += 1;
        } else {
          stableLetterRef.current = { letter: upper, count: 1 };
        }

        if (stableLetterRef.current.count >= STABLE_FRAMES_TO_COMMIT) {
          stableLetterRef.current = { letter: "", count: 0 };
          const tgt = targetRef.current;
          if (!tgt) return;
          setAttempts((a) => a + 1);
          if (upper === tgt) {
            setScore((s) => s + 1);
            setStreak((s) => {
              const ns = s + 1;
              setBestStreak((b) => (ns > b ? ns : b));
              return ns;
            });
            setRoundState("correct");
            setStatus(`✅ Correct! It was ${tgt}`);
            advanceTimeoutRef.current = window.setTimeout(() => {
              nextRound();
            }, ADVANCE_DELAY_MS);
          } else {
            setStreak(0);
            setStatus(`Saw ${upper}, keep trying`);
          }
        }
      } catch (e) {
        setStatus(e instanceof Error ? e.message : "classify error");
      } finally {
        inflightRef.current = false;
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [nextRound]);

  useEffect(() => {
    return () => {
      stopAudio();
      if (advanceTimeoutRef.current !== null) clearTimeout(advanceTimeoutRef.current);
    };
  }, [stopAudio]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      if (e.code === "KeyR") {
        e.preventDefault();
        replay();
      } else if (e.code === "KeyS") {
        e.preventDefault();
        skip();
      } else if (e.code === "KeyH") {
        e.preventDefault();
        reveal();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [replay, skip, reveal]);

  const accuracy = attempts > 0 ? Math.round((score / attempts) * 100) : 0;
  const isCorrect = roundState === "correct";

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Listening Practice</h1>
          <span className="tag">Translator mode · interpret the spoken letter into ASL</span>
        </div>
        <div style={{
          fontSize: '11px',
          fontWeight: 'bold',
          color: 'var(--accent-2)',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          background: 'rgba(34, 211, 238, 0.1)',
          padding: '4px 10px',
          borderRadius: '20px',
          border: '1px solid rgba(34, 211, 238, 0.3)'
        }}>
          ElevenLabs Voice
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Your Camera</h2>
          <Webcam onLandmarks={handleLandmarks} />
          <div style={{
            marginTop: 16,
            textAlign: 'center',
            padding: 12,
            borderRadius: 8,
            background: isCorrect ? '#065f46' : '#1e293b',
            border: `2px solid ${isCorrect ? '#4ade80' : '#334155'}`,
            transition: 'all 0.2s',
          }}>
            <div style={{ fontSize: 14, color: 'var(--muted)', marginBottom: 4 }}>Detected sign</div>
            <div style={{ fontSize: 48, fontWeight: 'bold' }}>{detected}</div>
            <div style={{ fontSize: 16, color: isCorrect ? '#4ade80' : 'var(--muted)' }}>
              {isCorrect ? '✨ MATCH ✨' : `confidence ${(confidence * 100).toFixed(0)}%`}
            </div>
          </div>
          <div className="status">{status}</div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ textAlign: 'center' }}>
            <h2>Incoming voice</h2>
            <div style={{
              padding: 24,
              borderRadius: 12,
              background: '#0a0f1e',
              border: '1px solid #2a3457',
              minHeight: 120,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 12,
            }}>
              {target ? (
                <>
                  <div style={{
                    fontSize: 18,
                    color: 'var(--muted)',
                    fontStyle: 'italic',
                  }}>
                    {roundState === 'playing' ? '🔊 Speaking…' : `"${phrase}"`}
                  </div>
                  <div style={{
                    fontSize: 96,
                    fontWeight: 'bold',
                    color: revealed || isCorrect ? 'var(--accent-2)' : '#1e293b',
                    textShadow: revealed || isCorrect ? '0 0 30px rgba(34,211,238,0.4)' : 'none',
                    letterSpacing: '8px',
                    transition: 'all 0.3s',
                  }}>
                    {revealed || isCorrect ? target : '?'}
                  </div>
                </>
              ) : (
                <div style={{ color: 'var(--muted)', fontSize: 16 }}>
                  Press <strong>Start</strong> and a "client" will speak a letter for you to sign.
                </div>
              )}
            </div>
            <div className="row" style={{ justifyContent: 'center', marginTop: 16, flexWrap: 'wrap' }}>
              {!target ? (
                <button onClick={nextRound} style={{ padding: '10px 24px', fontSize: 16 }}>
                  ▶ Start
                </button>
              ) : (
                <>
                  <button onClick={replay} className="secondary" disabled={roundState === 'playing'}>
                    🔁 Replay [R]
                  </button>
                  <button onClick={reveal} className="secondary" disabled={revealed || isCorrect}>
                    👁 Reveal [H]
                  </button>
                  <button onClick={skip} className="secondary">
                    ⏭ Skip [S]
                  </button>
                  <button onClick={nextRound} disabled={roundState === 'playing'}>
                    Next letter
                  </button>
                </>
              )}
            </div>
          </div>

          <div>
            <h2>Score</h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: 12,
              textAlign: 'center',
            }}>
              <Stat label="Correct" value={score} color="var(--accent-2)" />
              <Stat label="Streak" value={streak} color="#4ade80" />
              <Stat label="Best" value={bestStreak} color="var(--accent)" />
              <Stat label="Accuracy" value={`${accuracy}%`} color="#fbbf24" />
            </div>
            <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
              <button className="secondary" onClick={reset}>Reset session</button>
            </div>
          </div>

          <div style={{
            padding: 16,
            background: 'rgba(124, 92, 255, 0.08)',
            border: '1px solid rgba(124, 92, 255, 0.3)',
            borderRadius: 12,
            fontSize: 13,
            color: 'var(--muted)',
            lineHeight: 1.5,
          }}>
            <strong style={{ color: 'var(--text)' }}>How it works.</strong> A spoken
            client will say a letter. Hold its ASL sign in frame until the
            classifier locks it in. The round auto-advances on a match.
            Letters J and Z are excluded (they require motion).
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div style={{
      padding: 12,
      borderRadius: 8,
      background: '#0a0f1e',
      border: '1px solid #2a3457',
    }}>
      <div style={{ fontSize: 12, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 'bold', color, marginTop: 4 }}>
        {value}
      </div>
    </div>
  );
}
