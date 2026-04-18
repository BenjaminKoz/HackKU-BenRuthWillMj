import type { Landmark } from "../hooks/useHandLandmarker";

export type ClassifyMode = "letters" | "words";

export async function classify(
  landmarks: Landmark[],
  mode: ClassifyMode = "letters",
): Promise<{ letter: string; confidence: number }> {
  const res = await fetch("/api/classify", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ landmarks, mode }),
  });
  if (!res.ok) throw new Error(`classify failed: ${res.status}`);
  return res.json();
}

export async function resetClassify(): Promise<void> {
  await fetch("/api/classify/reset", { method: "POST" });
}

export async function fetchSupportedWords(): Promise<string[]> {
  const res = await fetch("/api/words");
  if (!res.ok) throw new Error(`fetchSupportedWords failed: ${res.status}`);
  const data = await res.json();
  return (data.words ?? []) as string[];
}

export async function classifyWordClip(
  frames: Landmark[][][],
): Promise<{ letter: string; confidence: number }> {
  const res = await fetch("/api/classify-word-clip", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  if (!res.ok) throw new Error(`classify-word-clip failed: ${res.status}`);
  return res.json();
}

export async function compose(letters: string, mode: ClassifyMode = "letters"): Promise<string> {
  const res = await fetch("/api/compose", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ letters, mode }),
  });
  if (!res.ok) throw new Error(`compose failed: ${res.status}`);
  const data = await res.json();
  return data.text as string;
}

export async function speak(text: string): Promise<Blob> {
  const res = await fetch("/api/speak", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`speak failed: ${res.status}`);
  return res.blob();
}

export async function generateSentence(): Promise<{ sentence_with_blank: string; target_word: string }> {
  const res = await fetch("/api/sentence-game/generate");
  if (!res.ok) throw new Error(`generateSentence failed: ${res.status}`);
  return res.json();
}

export async function validateWord(sentence_with_blank: string, user_word: string): Promise<{ is_correct: boolean; explanation: string }> {
  const res = await fetch("/api/sentence-game/validate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ sentence_with_blank, user_word }),
  });
  if (!res.ok) throw new Error(`validateWord failed: ${res.status}`);
  return res.json();
}
