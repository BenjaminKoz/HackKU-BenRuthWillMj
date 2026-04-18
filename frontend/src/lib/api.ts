import type { Landmark } from "../hooks/useHandLandmarker";

export type ClassifyMode = "letters" | "words";

// In dev, leave VITE_API_BASE_URL unset so fetches are relative and Vite's
// proxy forwards /api/* to localhost:8000. In prod (Vercel), set it to the
// deployed backend origin (e.g. https://<user>-<space>.hf.space).
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const url = (path: string) => `${API_BASE}${path}`;

export async function classify(
  landmarks: Landmark[],
  mode: ClassifyMode = "letters",
): Promise<{ letter: string; confidence: number }> {
  const res = await fetch(url("/api/classify"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ landmarks, mode }),
  });
  if (!res.ok) throw new Error(`classify failed: ${res.status}`);
  return res.json();
}

export async function resetClassify(): Promise<void> {
  await fetch(url("/api/classify/reset"), { method: "POST" });
}

export async function fetchSupportedWords(): Promise<string[]> {
  const res = await fetch(url("/api/words"));
  if (!res.ok) throw new Error(`fetchSupportedWords failed: ${res.status}`);
  const data = await res.json();
  return (data.words ?? []) as string[];
}

export async function classifyWordClip(
  frames: Landmark[][][],
): Promise<{ letter: string; confidence: number }> {
  const res = await fetch(url("/api/classify-word-clip"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ frames }),
  });
  if (!res.ok) throw new Error(`classify-word-clip failed: ${res.status}`);
  return res.json();
}

export async function compose(letters: string, mode: ClassifyMode = "letters"): Promise<string> {
  const res = await fetch(url("/api/compose"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ letters, mode }),
  });
  if (!res.ok) throw new Error(`compose failed: ${res.status}`);
  const data = await res.json();
  return data.text as string;
}

export async function speak(text: string): Promise<Blob> {
  const res = await fetch(url("/api/speak"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`speak failed: ${res.status}`);
  return res.blob();
}
