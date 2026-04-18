import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify } from "./lib/api";

const LETTERS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
  "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
  "U", "V", "W", "X", "Y", "Z"
];

const DESCRIPTIONS: Record<string, string> = {
  A: "Make a fist and place your thumb against the side of your index finger.",
  B: "Hold your hand up with your fingers straight and together. Fold your thumb across your palm.",
  C: "Curve your hand and fingers to form the shape of the letter 'C'.",
  D: "Touch your thumb to your middle, ring, and pinky fingers. Point your index finger straight up.",
  E: "Curl your fingers tightly into your palm and fold your thumb to rest across your fingers.",
  F: "Touch the tips of your index finger and thumb together. Keep your other three fingers straight and spread apart.",
  G: "Point your index finger forward and your thumb parallel to it, as if pinching something. Tuck other fingers in.",
  H: "Point your index and middle fingers forward, keeping them together. Tuck your thumb and other fingers in.",
  I: "Hold up your pinky finger. Fold all other fingers into a fist.",
  J: "Hold up your pinky finger (like 'I') and trace the shape of a 'J' in the air.",
  K: "Point your index and middle fingers upward in a 'V' shape. Place your thumb against the base of your index and middle fingers.",
  L: "Point your index finger straight up and your thumb straight out to the side to form an 'L' shape.",
  M: "Make a fist and tuck your thumb between your ring and pinky fingers.",
  N: "Make a fist and tuck your thumb between your middle and ring fingers.",
  O: "Curve all your fingers to touch the tip of your thumb, forming an 'O' shape.",
  P: "Point your index finger forward, drop your middle finger down, and place your thumb on the middle finger (like an upside-down 'K').",
  Q: "Point your index finger and thumb downward, as if holding something small (like an upside-down 'G').",
  R: "Cross your index and middle fingers, keeping them pointed up. Fold the other fingers into a fist.",
  S: "Make a fist and place your thumb over the front of your fingers.",
  T: "Make a fist and tuck your thumb between your index and middle fingers.",
  U: "Point your index and middle fingers straight up and keep them together. Fold other fingers into a fist.",
  V: "Point your index and middle fingers up and spread them apart (like a peace sign). Fold other fingers into a fist.",
  W: "Point your index, middle, and ring fingers up and spread them apart. Hold your pinky down with your thumb.",
  X: "Make a fist, then raise and hook your index finger (like a hook).",
  Y: "Extend your thumb and pinky fingers out to the sides. Fold the other three fingers into your palm.",
  Z: "Extend your index finger and trace the shape of a 'Z' in the air."
};

export function Learning() {
  const [mode, setMode] = useState<"learn" | "test">("learn");
  const [activeLetter, setActiveLetter] = useState("A");
  const [testTarget, setTestTarget] = useState("");
  const [score, setScore] = useState(0);
  
  // Real-time feedback state
  const [detectedLetter, setDetectedLetter] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [isCorrect, setIsCorrect] = useState(false);
  const [status, setStatus] = useState("Waiting for hand…");

  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const inflightRef = useRef(false);

  const startTest = useCallback(() => {
    setMode("test");
    const next = LETTERS[Math.floor(Math.random() * LETTERS.length)];
    setTestTarget(next);
    setScore(0);
  }, []);

  const nextTestItem = useCallback(() => {
    const next = LETTERS[Math.floor(Math.random() * LETTERS.length)];
    setTestTarget(next);
  }, []);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      const lms = lastLandmarksRef.current;
      const now = performance.now();
      if (!lms || inflightRef.current || now - lastClassifyRef.current < 250) return;
      
      lastClassifyRef.current = now;
      inflightRef.current = true;
      try {
        const { letter, confidence: conf } = await classify(lms);
        const upper = letter.toUpperCase();
        setDetectedLetter(upper);
        setConfidence(conf);
        
        const target = mode === "learn" ? activeLetter : testTarget;
        const correct = upper === target && conf > 0.65;
        setIsCorrect(correct);

        if (mode === "test" && correct) {
          // Auto-advance in test mode after a short delay
          setTimeout(() => {
            setScore(s => s + 1);
            nextTestItem();
          }, 1000);
        }
        setStatus("Tracking");
      } catch (e) {
        setStatus("Classifier unavailable");
      } finally {
        inflightRef.current = false;
      }
    }, 100);
    return () => clearInterval(id);
  }, [mode, activeLetter, testTarget, nextTestItem]);

  return (
    <div className="app">
      <div className="header">
        <h1>{mode === "learn" ? "Learning ASL" : "ASL Skill Test"}</h1>
        <div className="row">
           <button onClick={() => setMode("learn")} className={mode === "learn" ? "" : "secondary"}>Study Guide</button>
           <button onClick={startTest} className={mode === "test" ? "" : "secondary"}>Test Me!</button>
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Practice Area</h2>
          <Webcam onLandmarks={handleLandmarks} />
          
          <div style={{ marginTop: '16px', textAlign: 'center' }}>
            <div style={{ 
              padding: '12px', 
              borderRadius: '8px', 
              background: isCorrect ? '#065f46' : '#1e293b',
              border: `2px solid ${isCorrect ? '#4ade80' : '#334155'}`,
              transition: 'all 0.2s'
            }}>
              <div style={{ fontSize: '14px', color: 'var(--muted)', marginBottom: '4px' }}>Detected Sign:</div>
              <div style={{ fontSize: '48px', fontWeight: 'bold' }}>{detectedLetter}</div>
              <div style={{ fontSize: '18px', color: isCorrect ? '#4ade80' : 'var(--muted)' }}>
                {isCorrect ? "✨ CORRECT! ✨" : `Confidence: ${(confidence * 100).toFixed(0)}%`}
              </div>
            </div>
          </div>
          <div className="status">{status}</div>
        </div>

        <div className="panel">
          {mode === "learn" ? (
            <>
              <h2>Select a letter to learn</h2>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '20px' }}>
                {LETTERS.map(l => (
                  <button 
                    key={l} 
                    onClick={() => setActiveLetter(l)}
                    className={l === activeLetter ? "" : "secondary"}
                    style={{ width: '40px', height: '40px', padding: 0 }}
                  >
                    {l}
                  </button>
                ))}
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '120px', color: 'var(--accent)', margin: '20px 0' }}>{activeLetter}</div>
                <h2>Instructions</h2>
                <div className="sentence">{DESCRIPTIONS[activeLetter]}</div>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <h2>Challenge: Sign this letter</h2>
              <div style={{ fontSize: '160px', color: 'var(--accent-2)', margin: '40px 0', textShadow: '0 0 30px rgba(34,211,238,0.3)' }}>
                {testTarget}
              </div>
              <div style={{ fontSize: '24px', marginBottom: '20px' }}>
                Score: <span style={{ color: 'var(--accent-2)', fontWeight: 'bold' }}>{score}</span>
              </div>
              <p style={{ color: 'var(--muted)' }}>Hold the sign correctly for 1 second to score!</p>
              <button className="secondary" onClick={() => setMode("learn")} style={{ marginTop: '20px' }}>Exit Test</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
