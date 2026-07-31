// Landing page: list sample markets and let a visitor run the model.
// Phase 1 reads precomputed expert probabilities bundled with each market;
// the "Predict" action (MarketCard) calls the API's MoE gate.

import { listMarkets, type Market } from "../lib/api";
import MarketCard from "./MarketCard";

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
              <MarketCard key={m.id} market={m} />
            ))}
          </div>
        )}
      </section>
    </>
  );
}
