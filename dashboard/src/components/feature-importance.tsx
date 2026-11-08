import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export function FeatureImportance({ shap }: { shap: Record<string, number> | undefined }) {
  const entries = shap ? Object.entries(shap).slice(0, 8) : [];
  const max = entries[0]?.[1] || 1;
  return (
    <Card>
      <CardHeader>
        <CardTitle>What drives the score</CardTitle>
        <CardDescription>Mean |SHAP| on the out-of-time test set</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {entries.map(([name, v]) => (
          <div key={name} className="grid grid-cols-[9.5rem_1fr_3rem] items-center gap-2 text-xs">
            <span className="truncate font-mono text-muted-foreground">{name}</span>
            <div className="h-2 rounded-full bg-muted">
              <div className="h-2 rounded-full bg-primary" style={{ width: `${(v / max) * 100}%` }} />
            </div>
            <span className="text-right tabular-nums">{v.toFixed(3)}</span>
          </div>
        ))}
        {!entries.length && <p className="text-sm text-muted-foreground">Waiting for model metadata…</p>}
      </CardContent>
    </Card>
  );
}
