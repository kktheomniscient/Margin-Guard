"use client";

import { useCallback, useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { DriftMonitor } from "@/components/drift-monitor";
import { FeatureImportance } from "@/components/feature-importance";
import { KpiBanner } from "@/components/kpi-banner";
import { RiskSimulator } from "@/components/risk-simulator";
import { api, ApiError, type DriftStatus, type ModelInfo } from "@/lib/api";

export default function RiskConsole() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [drift, setDrift] = useState<DriftStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refreshInfo = useCallback(() => {
    api
      .modelInfo()
      .then((i) => {
        setInfo(i);
        setError(null);
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Unexpected error"));
  }, []);

  useEffect(refreshInfo, [refreshInfo]);

  return (
    <main className="mx-auto w-full max-w-7xl space-y-6 p-4 sm:p-8">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b pb-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">MarginGuard</h1>
          <p className="text-sm text-muted-foreground">
            Market-risk lakehouse &amp; tail-risk model operations · Bronze → Silver → Gold → LightGBM
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={error ? "destructive" : "secondary"}>{error ? "API offline" : "API online"}</Badge>
          {info && (
            <a href={info.mlflow_url} target="_blank" rel="noreferrer">
              <Badge variant="outline">
                MLflow <ExternalLink />
              </Badge>
            </a>
          )}
        </div>
      </header>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Backend unavailable</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <KpiBanner info={info} drift={drift} />
      <RiskSimulator
        onScored={() => {
          refreshInfo();
          setTick((t) => t + 1);
        }}
      />
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <div className="xl:col-span-3">
          <DriftMonitor onLoaded={setDrift} refreshKey={tick} />
        </div>
        <div className="xl:col-span-2">
          <FeatureImportance shap={info?.shap_importance} />
        </div>
      </div>
    </main>
  );
}
