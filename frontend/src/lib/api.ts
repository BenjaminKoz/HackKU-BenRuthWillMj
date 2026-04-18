import type { Landmark } from "../hooks/useHandLandmarker";

export async function classify(landmarks: Landmark[]): Promise<{ letter: string; confidence: number }> {
  const res = await fetch("/api/classify", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ landmarks }),
  });
  if (!res.ok) throw new Error(`classify failed: ${res.status}`);
  return res.json();
}

export async function compose(letters: string): Promise<string> {
  const res = await fetch("/api/compose", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ letters }),
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
