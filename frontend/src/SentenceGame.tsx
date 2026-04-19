import { useCallback, useEffect, useRef, useState } from "react";
import { Webcam } from "./components/Webcam";
import type { Landmark } from "./hooks/useHandLandmarker";
import { classify } from "./lib/api";
import { generateSentence, validateWord } from "./lib/sentences";

const CLASSIFY_INTERVAL_MS = 300;
const STABLE_FRAMES_TO_COMMIT = 5;
const STABLE_FRAMES_TO_SUGGEST = 8;
const MIN_CONFIDENCE = 0.60;
const SUGGESTION_THRESHOLD = 0.40;

export function SentenceGame() {
  const [sentence, setSentence] = useState("");
  const [possibleWords, setPossibleWords] = useState<string[]>([]);
  const [buffer, setBuffer] = useState("");
  const [validating, setValidating] = useState(false);
  const [result, setResult] = useState<{ is_correct: boolean; explanation: string } | null>(null);
  
  // ASL State
  const [currentLetter, setCurrentLetter] = useState<string>("-");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Waiting for hand…");
  const [suggestion, setSuggestion] = useState<string | null>(null);
  
  const lastLandmarksRef = useRef<Landmark[] | null>(null);
  const lastClassifyRef = useRef(0);
  const stableLetterRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const stableSuggestionRef = useRef<{ letter: string; count: number }>({ letter: "", count: 0 });
  const inflightRef = useRef(false);

  const initGame = useCallback(() => {
    setResult(null);
    setBuffer("");
    setSuggestion(null);
    const data = generateSentence();
    setSentence(data.sentence_with_blank);
    setPossibleWords(data.possible_words);
    setStatus("Waiting for hand…");
  }, []);

  useEffect(() => {
    initGame();
  }, [initGame]);

  const handleLandmarks = useCallback((lms: Landmark[] | null) => {
    lastLandmarksRef.current = lms;
  }, []);

  const onBackspace = () => setBuffer(b => b.slice(0, -1));
  const onClear = () => setBuffer("");

  const onCheck = () => {
    if (!buffer || validating) return;
    setValidating(true);
    setResult(validateWord(sentence, buffer));
    setValidating(false);
  };

  useEffect(() => {
    const id = setInterval(async () => {
      if (validating || result?.is_correct) return;
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
            setBuffer(b => b + upperLetter);
            stableLetterRef.current = { letter: "", count: 0 };
          }
          setSuggestion(null);
          stableSuggestionRef.current = { letter: "", count: 0 };
        } else if (conf >= SUGGESTION_THRESHOLD && upperLetter !== "NOTHING") {
          const prevS = stableSuggestionRef.current;
          if (prevS.letter === upperLetter) {
            prevS.count += 1;
          } else {
            stableSuggestionRef.current = { letter: upperLetter, count: 1 };
          }

          if (stableSuggestionRef.current.count >= STABLE_FRAMES_TO_SUGGEST) {
            setSuggestion(upperLetter);
          }
          stableLetterRef.current = { letter: "", count: 0 };
        } else {
          setSuggestion(null);
          stableLetterRef.current = { letter: "", count: 0 };
          stableSuggestionRef.current = { letter: "", count: 0 };
        }
        setStatus("Tracking");
      } catch (e) {
        setStatus("ASL Recognition Error");
      } finally {
        inflightRef.current = false;
      }
    }, 100);
    return () => clearInterval(id);
  }, [validating, result]);

  const handleSuggestionClick = () => {
    if (suggestion) {
      setBuffer(b => b + suggestion);
      setSuggestion(null);
      stableSuggestionRef.current = { letter: "", count: 0 };
    }
  };

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>Sentence Fill-in</h1>
          <span className="tag">Sign the letters to fill the blank!</span>
        </div>
        <div style={{
          fontSize: '11px',
          fontWeight: 'bold',
          color: 'var(--accent-2)',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          background: 'rgba(34, 211, 238, 0.1)',
          padding: '4px 10px',
          borderRadius: '20px',
          border: '1px solid rgba(34, 211, 238, 0.3)'
        }}>
          Interactive Learning
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          {suggestion && (
            <div style={{
              marginBottom: '12px',
              padding: '10px',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              border: '2px dashed #3b82f6',
              borderRadius: '12px',
              textAlign: 'center',
              animation: 'pulse 2s infinite'
            }}>
              <span style={{ fontSize: '14px' }}>Could it be <strong>{suggestion}</strong>?</span>
              <button 
                onClick={handleSuggestionClick}
                style={{ marginLeft: '10px', padding: '4px 12px', fontSize: '12px' }}
              >
                Add {suggestion}
              </button>
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
                <h2>Complete the sentence</h2>
                <div style={{ fontSize: '24px', margin: '20px 0', lineHeight: '1.5' }}>
                  {sentence.split("___").map((part, i, arr) => (
                    <span key={i}>
                      {part}
                      {i < arr.length - 1 && (
                        <span style={{ 
                          borderBottom: '2px solid var(--accent-2)', 
                          color: 'var(--accent-2)',
                          minWidth: '60px',
                          display: 'inline-block',
                          padding: '0 8px'
                        }}>
                          {buffer || "___"}
                        </span>
                      )}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h2>Your Answer (ASL Input)</h2>
                <div className="buffer" style={{ fontSize: '32px', textAlign: 'center', letterSpacing: '4px' }}>
                  {buffer || <span style={{ color: '#4b5563' }}>START SIGNING...</span>}
                </div>
                <div style={{ marginTop: '16px' }}>
                  <div style={{ fontSize: '13px', color: 'var(--muted)', marginBottom: '10px', textAlign: 'center' }}>
                    Possible words
                  </div>
                  <div className="row" style={{ flexWrap: 'wrap', justifyContent: 'center', gap: '10px' }}>
                    {possibleWords.map((word) => (
                      <button
                        key={word}
                        className="secondary"
                        onClick={() => setBuffer(word)}
                        type="button"
                        style={{
                          fontSize: '13px',
                          letterSpacing: '1px',
                          padding: '8px 12px',
                          opacity: buffer === word ? 1 : 0.85,
                        }}
                      >
                        {word}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="row" style={{ justifyContent: 'center', marginTop: '16px' }}>
                  <button className="secondary" onClick={onBackspace}>Backspace</button>
                  <button className="secondary" onClick={onClear}>Clear</button>
                  <button onClick={onCheck} disabled={!buffer || validating}>
                    {validating ? "Checking..." : "Check Answer"}
                  </button>
                </div>
              </div>

              {result && (
                <div style={{
                  padding: '20px',
                  borderRadius: '12px',
                  backgroundColor: result.is_correct ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                  border: `2px solid ${result.is_correct ? '#10b981' : '#ef4444'}`,
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '8px' }}>
                    {result.is_correct ? "✅ Correct!" : "❌ Not quite"}
                  </div>
                  <div style={{ fontSize: '16px' }}>{result.explanation}</div>
                  {result.is_correct && (
                    <button onClick={initGame} style={{ marginTop: '16px' }}>Next Sentence</button>
                  )}
                </div>
              )}
          <button className="secondary" onClick={initGame} style={{ marginTop: 'auto' }}>
            New Sentence
          </button>
        </div>
      </div>
    </div>
  );
}
