# Web frontend

Next.js (App Router) UI for the demo — deploys to **Vercel free tier**. Styling
is a Kalshi-inspired but original look (no Kalshi logos/marks; disclaimer in the
footer).

```
web/
├── app/
│   ├── layout.tsx      # shell, header, footer disclaimer
│   ├── page.tsx        # market list + predict wiring
│   └── globals.css     # Kalshi-ish palette
└── lib/api.ts          # client for the inference API
```

## Local dev

```bash
npm install
cp .env.example .env.local   # point NEXT_PUBLIC_API_URL at the API
npm run dev
```

## Deploy

Import `app/web` as the Vercel project root. Set `NEXT_PUBLIC_API_URL` to the
deployed inference API URL in Vercel env vars.

The predict button loads a bundled sample market from the API and posts its
gate features plus expert probabilities to `/predict`.
