import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify } from "./lib/api";

const WORDS = ["APPLE", "BANANA", "CHERRY", "DRAGON", "EAGLE", "FLOWER", "GALAXY", "GUITAR", "HAMMER", "ISLAND", "JACKET", "KETTLE", "LANTERN", "MOUNTAIN", "NEBULA", "OCEAN", "PIANO", "QUARTZ", "RABBIT", "SILVER", "TIGER", "UMBRELLA", "VALLEY", "WINTER", "XYLOPHONE", "YELLOW", "ZEBRA"];
const MAX_LIVES = 6;
const CLASSIFY_INTERVAL_MS = 300;
const STABLE_FRAMES_TO_COMMIT = 5;
const STABLE_FRAMES_TO_SUGGEST = 8; // ~2.4s of stability
const MIN_CONFIDENCE = 0.60;
const SUGGESTION_THRESHOLD = 0.40;

export function Game() {
  const [word, setWord] = useState("");
  const [guessed, setGuessed] = useState<string[]>([]);
  const [lives, setLives] = useState(MAX_LIVES);
  const [gameOver, setGameOver] = useState(false);
  const [won, setWon] = useState(false);
  
  // ASL State
  const [currentLetter, setCurrentLetter] = useState<string>("-");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Waiting for hand…");
  const [suggestion, setSuggestion] = useState<string | null>(null);
  const [confirmLetter, setConfirmLetter] = useState<string | null>(null);
  const [gameHint, setGameHint] = useState<{ letter: string; description: string } | null>(null);
  
  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const stableSuggestionRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const inflightRef = useRef(false);

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

  const initGame = useCallback(() => {
    const w = WORDS[Math.floor(Math.random() * WORDS.length)];
    setWord(w);
    setGuessed([]);
    setLives(MAX_LIVES);
    setGameOver(false);
    setWon(false);
    setSuggestion(null);
    setConfirmLetter(null);
    setGameHint(null);
    stableLetterRef.current = { letter: "", count: 0 };
    stableSuggestionRef.current = { letter: "", count: 0 };
  }, []);

  useEffect(() => {
    initGame();
  }, [initGame]);

  const makeGuess = useCallback((letter: string) => {
    if (gameOver || won || guessed.includes(letter)) {
      setConfirmLetter(null);
      return;
    }
    
    const newGuessed = [...guessed, letter];
    setGuessed(newGuessed);
    setSuggestion(null);
    setConfirmLetter(null);
    setGameHint(null);
    stableSuggestionRef.current = { letter: "", count: 0 };
    
    if (!word.includes(letter)) {
      const newLives = lives - 1;
      setLives(newLives);
      if (newLives <= 0) setGameOver(true);
    } else {
      const allGuessed = word.split("").every(l => newGuessed.includes(l));
      if (allGuessed) setWon(true);
    }
  }, [word, guessed, lives, gameOver, won]);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
  }, []);

  useEffect(() => {
    const id = setInterval(async () => {
      if (gameOver || won) return;
      const lms = lastLandmarksRef.current;
      const now = performance.now();
      if (!lms || inflightRef.current || now - lastClassifyRef.current < CLASSIFY_INTERVAL_MS) return;
      
      lastClassifyRef.current = now;
      inflightRef.current = true;
      try {
        const { letter, confidence: conf } = await classify(lms);
        const upperLetter = letter.toUpperCase();
        setCurrentLetter(upperLetter);
        setConfidence(conf);
        
        if (conf >= MIN_CONFIDENCE && upperLetter !== "NOTHING" && !guessed.includes(upperLetter)) {
          // Stability check for confirmation
          const prev = stableLetterRef.current;
          if (prev.letter === upperLetter) {
            prev.count += 1;
          } else {
            stableLetterRef.current = { letter: upperLetter, count: 1 };
          }
          
          if (stableLetterRef.current.count === STABLE_FRAMES_TO_COMMIT) {
            setConfirmLetter(upperLetter);
            setGameHint(null); // Clear manual hint if they actually sign it
            stableLetterRef.current = { letter: "", count: 0 };
          }
          
          // Reset suggestion tracking
          setSuggestion(null);
          stableSuggestionRef.current = { letter: "", count: 0 };

        } else if (conf >= SUGGESTION_THRESHOLD && upperLetter !== "NOTHING" && !guessed.includes(upperLetter) && !confirmLetter) {
          // Suggestion logic with stability check
          const prevS = stableSuggestionRef.current;
          if (prevS.letter === upperLetter) {
            prevS.count += 1;
          } else {
            stableSuggestionRef.current = { letter: upperLetter, count: 1 };
          }

          if (stableSuggestionRef.current.count >= STABLE_FRAMES_TO_SUGGEST) {
            setSuggestion(upperLetter);
          }
          
          // Reset auto-guess tracking
          stableLetterRef.current = { letter: "", count: 0 };

        } else {
          // Reset everything if confidence is too low or "NOTHING"
          // We don't reset confirmLetter here, user must interact with UI
          stableLetterRef.current = { letter: "", count: 0 };
          stableSuggestionRef.current = { letter: "", count: 0 };
          if (upperLetter === "NOTHING" || conf < SUGGESTION_THRESHOLD) {
             setSuggestion(null);
          }
        }
        setStatus("Tracking");
      } catch (e) {
        setStatus("ASL Recognition Error (Model Missing?)");
      } finally {
        inflightRef.current = false;
      }
    }, 100);
    return () => clearInterval(id);
  }, [makeGuess, gameOver, won, guessed, confirmLetter]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (gameOver || won) return;

      if (e.code === "Space") {
        e.preventDefault(); // Prevent scrolling
        if (confirmLetter) {
          makeGuess(confirmLetter);
        } else if (suggestion) {
          makeGuess(suggestion);
        } else if (gameHint) {
          makeGuess(gameHint.letter);
        }
      } else if (e.code === "Escape") {
        if (confirmLetter) {
          setConfirmLetter(null);
          stableLetterRef.current = { letter: "", count: 0 };
        } else if (suggestion) {
          setSuggestion(null);
          stableSuggestionRef.current = { letter: "", count: 0 };
        } else if (gameHint) {
          setGameHint(null);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [confirmLetter, suggestion, gameHint, makeGuess, gameOver, won]);

  const displayWord = word.split("").map(l => (guessed.includes(l) ? l : "_")).join(" ");

  return (
    <div className="app">
      <div className="header">
        <h1>ASL Hangman</h1>
        <span className="tag">Sign a letter to guess!</span>
      </div>

      <div className="grid">
        <div className="panel">
          {gameHint && !confirmLetter && !gameOver && !won && (
            <div style={{
              marginBottom: '20px',
              padding: '20px',
              backgroundColor: 'rgba(124, 92, 255, 0.1)',
              border: '2px solid var(--accent)',
              borderRadius: '12px',
              textAlign: 'center',
              boxShadow: '0 0 20px rgba(124, 92, 255, 0.2)',
              animation: 'bounce 0.5s ease-in-out'
            }}>
               <p style={{ margin: '0 0 10px 0', fontSize: '18px', fontWeight: 'bold' }}>
                How to sign <span style={{ color: 'var(--accent)', fontSize: '24px' }}>{gameHint.letter}</span>:
              </p>
              <p style={{ margin: '0 0 20px 0', fontSize: '15px', color: 'var(--text)', lineHeight: '1.4' }}>
                {gameHint.description}
              </p>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button 
                   className="secondary"
                   onClick={() => setGameHint(null)}
                   style={{ padding: '8px 16px' }}
                >
                  Cancel [Esc]
                </button>
                <button 
                  onClick={() => makeGuess(gameHint.letter)}
                  style={{ padding: '8px 16px', backgroundColor: '#334155' }}
                >
                  Guess [Space]
                </button>
              </div>
            </div>
          )}

          {confirmLetter && !gameOver && !won && (
            <div style={{
              marginBottom: '20px',
              padding: '20px',
              backgroundColor: 'rgba(34, 211, 238, 0.1)',
              border: '2px solid var(--accent-2)',
              borderRadius: '12px',
              textAlign: 'center',
              boxShadow: '0 0 20px rgba(34, 211, 238, 0.2)',
              animation: 'bounce 0.5s ease-in-out'
            }}>
              <p style={{ margin: '0 0 15px 0', fontSize: '18px', fontWeight: 'bold' }}>
                Confirm Guess: <span style={{ color: 'var(--accent-2)', fontSize: '28px' }}>{confirmLetter}</span>?
              </p>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button 
                  onClick={() => makeGuess(confirmLetter)}
                  style={{ padding: '10px 30px', fontSize: '16px' }}
                >
                  Yes [Space]
                </button>
                <button 
                  className="secondary"
                  onClick={() => {
                    setConfirmLetter(null);
                    stableLetterRef.current = { letter: "", count: 0 };
                  }}
                  style={{ padding: '10px 20px', fontSize: '16px' }}
                >
                  No [Esc]
                </button>
              </div>
            </div>
          )}

          {suggestion && !confirmLetter && !gameHint && !gameOver && !won && (
            <div style={{
              marginBottom: '20px',
              padding: '15px',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              border: '2px dashed #3b82f6',
              borderRadius: '12px',
              textAlign: 'center',
              animation: 'pulse 2s infinite'
            }}>
              <p style={{ margin: '0 0 10px 0', fontSize: '16px' }}>Could it be <strong>{suggestion}</strong>?</p>
              <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
                <button 
                  onClick={() => makeGuess(suggestion)}
                  style={{ padding: '8px 20px', fontSize: '14px' }}
                >
                  Yes [Space]
                </button>
                <button 
                  className="secondary"
                  onClick={() => {
                    setSuggestion(null);
                    stableSuggestionRef.current = { letter: "", count: 0 };
                  }}
                  style={{ padding: '8px 20px', fontSize: '14px' }}
                >
                  Dismiss [Esc]
                </button>
              </div>
            </div>
          )}

          <h2>Your Camera</h2>
          <Webcam onLandmarks={handleLandmarks} />
          <div className="letter-big">{currentLetter}</div>
          <div className="confidence">confidence {(confidence * 100).toFixed(0)}%</div>
          <div className="status">{status}</div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ textAlign: 'center' }}>
            <h2>Word to Guess</h2>
            <div style={{ fontSize: '48px', letterSpacing: '8px', margin: '20px 0', fontFamily: 'monospace' }}>
              {displayWord}
            </div>
            <div style={{ fontSize: '20px', color: won ? '#4ade80' : gameOver ? '#f87171' : 'var(--text)' }}>
              {won ? "🎉 YOU WON!" : gameOver ? `💀 GAME OVER! The word was ${word}` : `Lives: ${"❤️".repeat(lives)}`}
            </div>
          </div>

          <div>
            <h2>Guessed Letters</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {"ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map(l => {
                const isGuessed = guessed.includes(l);
                const isCorrect = isGuessed && word.includes(l);
                const isWrong = isGuessed && !word.includes(l);
                return (
                  <button
                    key={l}
                    onClick={() => {
                      if (!isGuessed && !gameOver && !won) {
                        setGameHint({ letter: l, description: DESCRIPTIONS[l] });
                        setConfirmLetter(null);
                        setSuggestion(null);
                      }
                    }}
                    disabled={isGuessed || gameOver || won}
                    className="secondary"
                    style={{
                      width: '36px',
                      height: '36px',
                      padding: 0,
                      backgroundColor: isCorrect ? '#065f46' : isWrong ? '#7f1d1d' : 'transparent',
                      color: isGuessed ? '#fff' : 'var(--text)',
                      borderColor: isGuessed ? 'transparent' : '#2a3457'
                    }}
                  >
                    {l}
                  </button>
                );
              })}
            </div>
          </div>

          <button onClick={initGame} style={{ marginTop: 'auto' }}>
            New Game
          </button>
        </div>
      </div>
    </div>
  );
}
