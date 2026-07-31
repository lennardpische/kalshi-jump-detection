"use client";

import { useState } from "react";
import { getMarket, predict, type Market, type Prediction } from "../lib/api";

type State =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; result: Prediction }
  | { status: "error"; message: string };

export default function MarketCard({ market }: { market: Market }) {
  const [state, setState] = useState<State>({ status: "idle" });

  async function runPredict() {
    setState({ status: "loading" });
    try {
      const detail = await getMarket(market.id);
      const result = await predict({
        horizon: detail.horizon,
        gate_feats: detail.gate_feats,
        expert_probs: detail.expert_probs,
      });
      setState({ status: "done", result });
    } catch (err) {
      setState({
        status: "error",
        message: err instanceof Error ? err.message : "Prediction failed",
      });
    }
  }

  return (
    <article className="card">
      <span className="tag">{market.category}</span>
      <h3>{market.title}</h3>
      <p className="muted">{market.description}</p>

      <button
        className="btn"
        onClick={runPredict}
        disabled={state.status === "loading"}
      >
        {state.status === "loading" ? "Predicting…" : "Predict"}
      </button>

      {state.status === "error" && <p className="error-text">{state.message}</p>}

      {state.status === "done" && (
        <div className="result">
          <p className={`prediction prediction-${state.result.prediction}`}>
            {state.result.prediction.toUpperCase()}
          </p>
          <ul className="prob-list">
            {Object.entries(state.result.probabilities).map(([cls, p]) => (
              <li key={cls}>
                <span className="prob-label">{cls}</span>
                <span className="prob-bar-track">
                  <span
                    className="prob-bar-fill"
                    style={{ width: `${Math.round(p * 100)}%` }}
                  />
                </span>
                <span className="prob-value">{Math.round(p * 100)}%</span>
              </li>
            ))}
          </ul>
          <details className="gate-weights">
            <summary>Expert gate weights</summary>
            <ul>
              {Object.entries(state.result.gate_weights).map(([expert, w]) => (
                <li key={expert}>
                  {expert}: {Math.round(w * 100)}%
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </article>
  );
}
