import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Jump Detection — Prediction Market Demo",
  description: "ML jump-detection demo over prediction-market trades.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <span className="brand">Edge<span className="brand-accent">Finder</span></span>
          <nav className="nav">
            <a href="#markets">Markets</a>
            <a href="https://github.com" target="_blank" rel="noreferrer">Paper</a>
          </nav>
        </header>
        <main className="container">{children}</main>
        <footer className="footer">
          Research demo. Not affiliated with, endorsed by, or connected to Kalshi.
          Predictions are illustrative and not financial advice.
        </footer>
      </body>
    </html>
  );
}
