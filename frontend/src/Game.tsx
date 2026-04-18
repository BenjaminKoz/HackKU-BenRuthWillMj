import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify } from "./lib/api";

const WORDS = ["APPLE", "BANANA", "CHERRY", "DRAGON", "EAGLE", "FLOWER", "GALAXY", "GUITAR", "HAMMER", "ISLAND", "JACKET", "KETTLE", "LANTERN", "MOUNTAIN", "NEBULA", "OCEAN", "PIANO", "QUARTZ", "RABBIT", "SILVER", "TIGER", "UMBRELLA", "VALLEY", "WINTER", "XYLOPHONE", "YELLOW", "ZEBRA"];
const MAX_LIVES = 6;
const CLASSIFY_INTERVAL_MS = 300;
const STABLE_FRAMES_TO_COMMIT = 5;
const MIN_CONFIDENCE = 0.75;

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
  
  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const inflightRef = useRef(false);

  const initGame = useCallback(() => {
    const w = WORDS[Math.floor(Math.random() * WORDS.length)];
    setWord(w);
    setGuessed([]);
    setLives(MAX_LIVES);
    setGameOver(false);
    setWon(false);
  }, []);

  useEffect(() => {
    initGame();
  }, [initGame]);

  const makeGuess = useCallback((letter: string) => {
    if (gameOver || won || guessed.includes(letter)) return;
    
    const newGuessed = [...guessed, letter];
    setGuessed(newGuessed);
    
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
        
        if (conf >= MIN_CONFIDENCE && upperLetter !== "NOTHING") {
          const prev = stableLetterRef.current;
          if (prev.letter === upperLetter) {
            prev.count += 1;
          } else {
            stableLetterRef.current = { letter: upperLetter, count: 1 };
          }
          
          if (stableLetterRef.current.count === STABLE_FRAMES_TO_COMMIT) {
            makeGuess(upperLetter);
            stableLetterRef.current = { letter: "", count: 0 }; // Reset after guess
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
  }, [makeGuess, gameOver, won]);

  const displayWord = word.split("").map(l => (guessed.includes(l) ? l : "_")).join(" ");

  return (
    <div className="app">
      <div className="header">
        <h1>ASL Hangman</h1>
        <span className="tag">Sign a letter to guess!</span>
      </div>

      <div className="grid">
        <div className="panel">
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
                    onClick={() => makeGuess(l)}
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
