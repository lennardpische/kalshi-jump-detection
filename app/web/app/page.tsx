// Landing page: list sample markets and let a visitor run the model.
// Phase 1 reads precomputed expert probabilities bundled with each market;
// the "Predict" action calls the API's MoE gate.
//
// This is a scaffold — wiring (state, the predict call, result display) is left
// as clearly-marked TODOs so the structure can be reviewed first.

import { listMarkets, type Market } from "../lib/api";

async function getMarkets(): Promise<Market[]> {
  try {
    return await listMarkets();
  } catch {
    return []; // API not up yet; render the empty state.
  }
}

export default async function Home() {
  const markets = await getMarkets();

  return (
    <>
      <section className="hero">
        <h1>Can microstructure predict the next move?</h1>
        <p>
          Pick a market, feed in recent trades, and a Mixture-of-Experts model
          predicts whether the price moves <b>down</b>, stays <b>flat</b>, or
          moves <b>up</b> over the next 5–60 minutes.
        </p>
      </section>

      <section id="markets">
        <h2>Sample markets</h2>
        {markets.length === 0 ? (
          <p className="muted">
            No markets loaded — start the API and refresh.
          </p>
        ) : (
          <div className="grid">
            {markets.map((m) => (
              <article key={m.id} className="card">
                <span className="tag">{m.category}</span>
                <h3>{m.title}</h3>
                <p className="muted">{m.description}</p>
                {/* TODO: button -> POST /predict -> show probabilities + gate weights */}
                <button className="btn" disabled>
                  Predict (coming soon)
                </button>
              </article>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
