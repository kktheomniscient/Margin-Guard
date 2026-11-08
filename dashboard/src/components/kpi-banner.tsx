"use client";

import { Activity, Gauge, ShieldAlert, Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { DriftStatus, ModelInfo } from "@/lib/api";

function Kpi({ title, icon: Icon, children }: { title: string; icon: typeof Activity; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Icon className="size-4" /> {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">{children}</CardContent>
    </Card>
  );
}

export function KpiBanner({ info, drift }: { info: ModelInfo | null; drift: DriftStatus | null }) {
  const t = info?.metrics.test;
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Kpi title="Active model" icon={Activity}>
        {info ? (
          <>
            <div className="text-xl font-semibold">LightGBM</div>
            <p className="text-xs text-muted-foreground">
              {info.best_iteration} trees · {info.n_features} features · run{" "}
              <span className="font-mono">{info.run_id.slice(0, 8)}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Trained {new Date(info.trained_at).toLocaleDateString()} · data to {info.data_as_of ?? "n/a"}
            </p>
          </>
        ) : (
          <Skeleton className="h-14 w-full" />
        )}
      </Kpi>

      <Kpi title="Out-of-time PR-AUC" icon={Target}>
        {t ? (
          <>
            <div className="text-xl font-semibold tabular-nums">{t.pr_auc.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">
              {t.pr_auc_lift.toFixed(2)}× random (base rate {(t.prevalence * 100).toFixed(1)}%) · ROC-AUC{" "}
              {t.roc_auc.toFixed(3)}
            </p>
          </>
        ) : (
          <Skeleton className="h-14 w-full" />
        )}
      </Kpi>

      <Kpi title="Inference latency" icon={Gauge}>
        {info ? (
          <>
            <div className="text-xl font-semibold tabular-nums">
              {info.latency_ms_p50 === null ? "—" : `${info.latency_ms_p50.toFixed(1)} ms`}
              <span className="ml-1 text-xs font-normal text-muted-foreground">p50</span>
            </div>
            <p className="text-xs text-muted-foreground">
              {info.latency_ms_p95 === null
                ? "Score a request to measure"
                : `p95 ${info.latency_ms_p95.toFixed(1)} ms · `}
              {info.requests_served} served · budget 50 ms
            </p>
          </>
        ) : (
          <Skeleton className="h-14 w-full" />
        )}
      </Kpi>

      <Kpi title="Feature drift" icon={ShieldAlert}>
        {drift ? (
          <>
            <div className="flex items-center gap-2 text-xl font-semibold">
              {drift.dataset_drift_detected ? "Drift detected" : "Stable"}
              <Badge variant={drift.dataset_drift_detected ? "destructive" : "secondary"}>
                {(drift.share_of_drifted_columns * 100).toFixed(0)}% of features
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Latest market window vs. training reference ({drift.n_current} vs {drift.n_reference} rows)
            </p>
          </>
        ) : (
          <Skeleton className="h-14 w-full" />
        )}
      </Kpi>
    </div>
  );
}
