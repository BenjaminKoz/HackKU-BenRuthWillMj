import { useState } from "react";

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
  const [activeLetter, setActiveLetter] = useState("A");

  return (
    <div className="app">
      <div className="header">
        <h1>Learning ASL</h1>
        <span className="tag">Master the alphabet</span>
      </div>

      <div className="grid">
        <div className="panel" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignContent: 'flex-start' }}>
          {LETTERS.map(letter => (
            <button
              key={letter}
              onClick={() => setActiveLetter(letter)}
              className={letter === activeLetter ? "" : "secondary"}
              style={{ width: '48px', height: '48px', fontSize: '20px' }}
            >
              {letter}
            </button>
          ))}
        </div>

        <div className="panel">
          <h2>How to sign "{activeLetter}"</h2>
          <div className="letter-big" style={{ fontSize: '120px', color: 'var(--accent)', margin: '40px 0' }}>
            {activeLetter}
          </div>
          <div className="sentence" style={{ textAlign: 'center', marginTop: '24px' }}>
            {DESCRIPTIONS[activeLetter]}
          </div>
          {["J", "Z"].includes(activeLetter) && (
            <div className="status" style={{ textAlign: 'center', marginTop: '16px', color: 'var(--accent-2)' }}>
              Note: This sign involves motion.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
