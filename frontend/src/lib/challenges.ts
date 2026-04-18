// Pre-baked ASL-gloss challenge sentences. Each `words` list is the sequence
// the user has to sign in order; `english` is what we speak / show on success.
// Keep entries small (2-4 signs). Filtered at runtime to only sentences whose
// words are all in the trained vocabulary, so adding a new sign won't break
// the challenge flow but dropping one will.

export type Challenge = {
  words: string[];
  english: string;
};

export const CHALLENGES: Challenge[] = [
  { words: ["I", "LOVE", "YOU"], english: "I love you." },
  { words: ["I", "WANT", "MORE", "PLEASE"], english: "I want more, please." },
  { words: ["THANKS", "FRIEND"], english: "Thanks, friend." },
  { words: ["I", "NEED", "HELP"], english: "I need help." },
  { words: ["HELLO", "FRIEND"], english: "Hello, friend!" },
  { words: ["SORRY", "I", "UNDERSTAND"], english: "Sorry — I understand." },
  { words: ["YES", "PLEASE"], english: "Yes, please." },
  { words: ["NO", "THANKS"], english: "No, thanks." },
  { words: ["I", "HAPPY"], english: "I'm happy." },
  { words: ["HOW", "YOU"], english: "How are you?" },
  { words: ["I", "WANT", "EAT"], english: "I want to eat." },
  { words: ["YOU", "GOOD", "FRIEND"], english: "You are a good friend." },
  { words: ["PLEASE", "HELP"], english: "Please, help." },
  { words: ["I", "UNDERSTAND", "YOU"], english: "I understand you." },
  { words: ["THANKS", "I", "HAPPY"], english: "Thanks — I'm happy." },
  { words: ["EAT", "MORE", "PLEASE"], english: "Eat more, please." },
  { words: ["NO", "I", "UNDERSTAND"], english: "No, I don't understand." },
  { words: ["I", "NEED", "MORE"], english: "I need more." },
  { words: ["HELLO", "I", "GOOD"], english: "Hello — I'm good." },
  { words: ["HELP", "PLEASE"], english: "Help, please." },
];

export function pickRandomChallenge(
  available: Set<string>,
  avoid?: Challenge,
): Challenge | null {
  const pool = CHALLENGES.filter(
    (c) => c.words.every((w) => available.has(w)) && c !== avoid,
  );
  if (pool.length === 0) return null;
  return pool[Math.floor(Math.random() * pool.length)];
}
