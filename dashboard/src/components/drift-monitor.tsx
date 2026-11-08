"use client";

import { useEffect, useState } from "react";
import { Info, RefreshCw } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, ApiError, type DriftStatus } from "@/lib/api";

const PSI_VARIANT = { stable: "secondary", warn: "outline", alert: "destructive" } as const;

export function DriftMonitor({ onLoaded, refreshKey }: { onLoaded: (d: DriftStatus | null) => void; refreshKey: number }) {
  const [source, setSource] = useState<"market" | "live">("market");
  const [data, setData] = useState<DriftStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api
      .drift(source)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setError(null);
        if (source === "market") onLoaded(d);
      })
      .catch((e) => {
        if (cancelled) return;
        setData(null);
        setError(e instanceof ApiError ? e.message : "Unexpected error");
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [source, onLoaded, refreshKey, reload]);

  const refresh = () => {
    setLoading(true);
    setReload((n) => n + 1);
  };
  const choose = (next: "market" | "live") => {
    if (next === source) return;
    setData(null);
    setError(null);
    setLoading(true);
    setSource(next);
  };

  const rows = data ? Object.entries(data.feature_details).sort((a, b) => a[1].p_value - b[1].p_value) : [];

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <CardTitle>Drift monitor</CardTitle>
            <CardDescription>
              Evidently KS test (drift if p &lt; 0.05) and Population Stability Index vs. the training set
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant={source === "market" ? "default" : "outline"} onClick={() => choose("market")}>
              Market data
            </Button>
            <Button size="sm" variant={source === "live" ? "default" : "outline"} onClick={() => choose("live")}>
              Live requests
            </Button>
            <Button size="sm" variant="ghost" aria-label="Refresh drift" onClick={refresh}>
              <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && (
          <Alert>
            <Info />
            <AlertTitle>{source === "live" ? "Not enough live traffic yet" : "Drift unavailable"}</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {loading && !data && <Skeleton className="h-64 w-full" />}
        {data && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Feature</TableHead>
                <TableHead className="text-right">KS p-value</TableHead>
                <TableHead className="text-right">PSI</TableHead>
                <TableHead className="text-right">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(([name, d]) => (
                <TableRow key={name}>
                  <TableCell className="font-mono text-xs">{name}</TableCell>
                  <TableCell className="text-right tabular-nums">{d.p_value < 0.0001 ? "<0.0001" : d.p_value.toFixed(4)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    <Badge variant={PSI_VARIANT[d.psi_status]}>{d.psi.toFixed(2)}</Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge variant={d.drift_detected ? "destructive" : "secondary"}>
                      {d.drift_detected ? "Drifted" : "Stable"}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        {data && (
          <p className="text-xs text-muted-foreground">
            {data.n_current} current rows vs {data.n_reference} reference. Symbols are correlated, so KS is
            deliberately sensitive here — read it together with PSI (&lt;0.10 stable, &gt;0.25 major shift).
          </p>
        )}
      </CardContent>
    </Card>
  );
}
