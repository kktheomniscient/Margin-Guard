"use client";

import { useCallback, useEffect, useState } from "react";
import { Loader2, Zap } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, ApiError, FEATURES, type FeatureVector, type RiskPrediction } from "@/lib/api";

const TIER_STYLE = {
  LOW: { badge: "secondary", bar: "bg-emerald-500" },
  ELEVATED: { badge: "outline", bar: "bg-amber-500" },
  CRITICAL: { badge: "destructive", bar: "bg-red-500" },
} as const;

export function RiskSimulator({ onScored }: { onScored: () => void }) {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbol, setSymbol] = useState("");
  const [asOf, setAsOf] = useState<string | null>(null);
  const [form, setForm] = useState<Record<string, string>>({});
  const [result, setResult] = useState<RiskPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const loadSymbol = useCallback(async (s: string) => {
    setSymbol(s);
    setResult(null);
    try {
      const lf = await api.latestFeatures(s);
      setAsOf(lf.as_of);
      setForm(Object.fromEntries(FEATURES.map((f) => [f, String(Number(lf.features[f].toPrecision(5)))])));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load features");
    }
  }, []);

  useEffect(() => {
    api
      .symbols()
      .then((s) => {
        setSymbols(s);
        if (s.length) void loadSymbol(s[0]);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load symbols"));
  }, [loadSymbol]);

  async function score() {
    setBusy(true);
    setError(null);
    try {
      const features = Object.fromEntries(FEATURES.map((f) => [f, Number(form[f])])) as FeatureVector;
      if (Object.values(features).some((v) => Number.isNaN(v))) throw new ApiError(0, "All inputs must be numbers.");
      setResult(await api.predict(symbol, features));
      onScored();
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "Scoring failed");
    } finally {
      setBusy(false);
    }
  }

  const tier = result ? TIER_STYLE[result.risk_level] : null;

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <Card className="xl:col-span-3">
        <CardHeader>
          <CardTitle>Risk simulator</CardTitle>
          <CardDescription>
            Pick a symbol to load its latest engineered features{asOf ? ` (as of ${asOf})` : ""}, tweak them, and
            score a 3-session tail-risk event.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {symbols.map((s) => (
              <Button key={s} size="sm" variant={s === symbol ? "default" : "outline"} onClick={() => void loadSymbol(s)}>
                {s}
              </Button>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {FEATURES.map((f) => (
              <div key={f} className="space-y-1">
                <Label htmlFor={f} className="font-mono text-[11px] text-muted-foreground">
                  {f}
                </Label>
                <Input
                  id={f}
                  type="number"
                  step="any"
                  value={form[f] ?? ""}
                  onChange={(e) => setForm({ ...form, [f]: e.target.value })}
                />
              </div>
            ))}
          </div>
          <Button className="w-full" onClick={() => void score()} disabled={busy || !symbol || !Object.keys(form).length}>
            {busy ? <Loader2 className="animate-spin" /> : <Zap />} Run real-time inference
          </Button>
        </CardContent>
      </Card>

      <Card className="xl:col-span-2">
        <CardHeader>
          <CardTitle>Inference diagnostics</CardTitle>
          <CardDescription>Calibrated probability, not a raw model score</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {result && tier ? (
            <>
              <div>
                <div className="flex items-baseline justify-between">
                  <span className="text-sm text-muted-foreground">Probability of tail-risk event</span>
                  <span className="font-mono text-2xl font-semibold tabular-nums">{(result.risk_score * 100).toFixed(1)}%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-muted">
                  <div className={`h-2 rounded-full ${tier.bar}`} style={{ width: `${Math.min(result.risk_score * 100 * 3, 100)}%` }} />
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Historical base rate {(result.base_rate * 100).toFixed(1)}% → this is{" "}
                  {(result.risk_score / result.base_rate).toFixed(1)}× normal. Bar scaled ×3.
                </p>
              </div>
              <div className="flex items-center justify-between rounded-lg border p-3">
                <span className="text-sm">Risk tier</span>
                <Badge variant={tier.badge}>{result.risk_level}</Badge>
              </div>
              <div className="rounded-lg border p-3">
                <span className="mb-1 block text-xs text-muted-foreground">Recommended brokerage action</span>
                <span className="text-sm font-medium">{result.action_trigger}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Served in {result.latency_ms.toFixed(1)} ms · model {result.model_run_id.slice(0, 8)}
              </p>
            </>
          ) : (
            !error && <p className="text-sm text-muted-foreground">Run inference to see the model’s risk assessment.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
