# MarginGuard Console

Next.js (App Router) + Tailwind v4 + shadcn/ui front end for MarginGuard. See the [root README](../README.md) for the full project.

```bash
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev -- --port 3000
npm run lint && npm run build
```

- `src/lib/api.ts` – typed client for the FastAPI service
- `src/components/` – `kpi-banner`, `risk-simulator`, `drift-monitor`, `feature-importance`
- `src/components/ui/` – shadcn/ui primitives
