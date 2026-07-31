// Thin client for the inference API. Base URL comes from the env var so the
// same build works against localhost and the deployed Space/Render service.

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Market = {
  id: string;
  title: string;
  category: string;
  description: string;
};

export type MarketDetail = Market & {
  horizon: number;
  gate_feats: number[];
  expert_probs: Record<string, number[]>;
};

export type Prediction = {
  horizon: number;
  prediction: "down" | "flat" | "up";
  probabilities: Record<string, number>;
  gate_weights: Record<string, number>;
};

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export async function listMarkets(): Promise<Market[]> {
  const res = await fetch(`${API}/markets`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res, "Failed to load markets"));
  return res.json();
}

export async function getMarket(id: string): Promise<MarketDetail> {
  const res = await fetch(`${API}/markets/${id}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res, "Failed to load market"));
  return res.json();
}

export async function predict(body: unknown): Promise<Prediction> {
  const res = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readError(res, "Prediction failed"));
  return res.json();
}
