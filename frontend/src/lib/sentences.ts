type SentenceItem = {
  s: string;
  v: string[];
};

const SENTENCES: SentenceItem[] = [
  { s: "The dog was chasing the ___.", v: ["CAT", "BALL", "SQUIRREL", "CAR", "BIRD", "RABBIT"] },
  { s: "I like to eat a red ___.", v: ["APPLE", "CHERRY", "BERRY", "TOMATO", "FRUIT"] },
  { s: "The sun is a hot ___.", v: ["STAR", "BALL", "LIGHT", "FIRE", "CORE"] },
  { s: "Please open the ___.", v: ["DOOR", "WINDOW", "BOX", "BOOK", "JAR", "CAN"] },
  { s: "The blue ___ is in the sky.", v: ["BIRD", "PLANE", "KITE", "CLOUD", "MOON"] },
  { s: "I drink water from a ___.", v: ["CUP", "GLASS", "MUG", "BOTTLE", "BOWL"] },
  { s: "The cat sat on the ___.", v: ["MAT", "RUG", "COUCH", "CHAIR", "TABLE", "FLOOR", "LAP"] },
  { s: "She wears a hat on her ___.", v: ["HEAD", "HAIR", "FACE"] },
  { s: "I write with a pen on ___.", v: ["PAPER", "BOOK", "DESK", "PAGE"] },
  { s: "The fish swims in the ___.", v: ["WATER", "LAKE", "SEA", "OCEAN", "POND", "TANK"] },
  { s: "A big elephant has a long ___.", v: ["TRUNK", "NOSE", "TAIL"] },
  { s: "He drives a fast red ___.", v: ["CAR", "TRUCK", "VAN", "BIKE", "MOTOR"] },
  { s: "The tree has green ___.", v: ["LEAVES", "LEAF", "FRUIT", "BRANCH", "WOOD"] },
  { s: "I sleep in a warm ___.", v: ["BED", "ROOM", "HOUSE", "BAG", "TENT"] },
  { s: "The rain falls from the ___.", v: ["SKY", "CLOUD", "TOP"] },
  { s: "You use a key to open a ___.", v: ["LOCK", "DOOR", "GATE", "SAFE", "BOX"] },
  { s: "A monkey likes to eat a ___.", v: ["BANANA", "FRUIT", "NUT", "BERRY"] },
  { s: "I can see with my two ___.", v: ["EYES", "SIGHT", "LENS"] },
  { s: "The baker makes fresh ___.", v: ["BREAD", "CAKE", "PIE", "FOOD", "ROLLS"] },
  { s: "We play music on a ___.", v: ["PIANO", "DRUM", "GUITAR", "FLUTE", "HORN", "STAGE"] },
];

function shuffle<T>(arr: T[]): T[] {
  const copy = arr.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function generateSentence(): {
  sentence_with_blank: string;
  target_word: string;
  possible_words: string[];
} {
  const item = SENTENCES[Math.floor(Math.random() * SENTENCES.length)];
  return {
    sentence_with_blank: item.s,
    target_word: item.v[0],
    possible_words: shuffle(item.v),
  };
}

export function validateWord(
  sentence_with_blank: string,
  user_word: string,
): { is_correct: boolean; explanation: string } {
  const guess = user_word.toUpperCase().trim();
  const item = SENTENCES.find((it) => it.s === sentence_with_blank);

  if (!item) {
    const ok = guess.length >= 3;
    return {
      is_correct: ok,
      explanation: ok ? "That fits!" : "Try a longer word.",
    };
  }

  if (item.v.includes(guess)) {
    return {
      is_correct: true,
      explanation: `Yes! '${guess}' fits perfectly.`,
    };
  }

  const example = item.v[Math.floor(Math.random() * item.v.length)];
  return {
    is_correct: false,
    explanation: `Not quite. A word like '${example}' would fit well here.`,
  };
}
