// Typed fetch client for the MarginGuard FastAPI service.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const FEATURES = [
  "ret_1d", "ret_5d", "ret_20d", "ema_ratio", "ema_trend", "macd_norm", "macd_hist_norm",
  "rsi_14", "volatility_20d", "parkinson_vol_20d", "atr_ratio", "volume_zscore_20d",
  "abs_ret_z", "vol_ratio_5_20", "vol_ratio_20_60", "mkt_volatility_20d",
] as const;
export type FeatureName = (typeof FEATURES)[number];
export type FeatureVector = Record<FeatureName, number>;

export type RiskLevel = "LOW" | "ELEVATED" | "CRITICAL";

export interface RiskPrediction {
  symbol: string;
  risk_score: number;
  base_rate: number;
  risk_level: RiskLevel;
  action_trigger: string;
  latency_ms: number;
  model_run_id: string;
}

export interface SplitMetrics {
  pr_auc: number;
  pr_auc_lift: number;
  roc_auc: number;
  f1: number;
  brier: number;
  prevalence: number;
  alert_rate: number;
}

export interface ModelInfo {
  model_name: string;
  run_id: string;
  trained_at: string;
  n_features: number;
  best_iteration: number;
  thresholds: { elevated: number; critical: number };
  metrics: { validation: SplitMetrics; test: SplitMetrics };
  shap_importance: Record<string, number>;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  requests_served: number;
  mlflow_url: string;
  data_as_of: string | null;
}

export interface FeatureDrift {
  p_value: number;
  psi: number;
  drift_detected: boolean;
  psi_status: "stable" | "warn" | "alert";
}

export interface DriftStatus {
  source: "market" | "live";
  dataset_drift_detected: boolean;
  share_of_drifted_columns: number;
  drifted_features: string[];
  feature_details: Record<string, FeatureDrift>;
  n_reference: number;
  n_current: number;
  stattest: string;
}

export interface LatestFeatures {
  symbol: string;
  as_of: string;
  features: FeatureVector;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is the FastAPI service running?`);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  modelInfo: () => request<ModelInfo>("/api/v1/model-info"),
  drift: (source: "market" | "live") => request<DriftStatus>(`/api/v1/drift-status?source=${source}`),
  symbols: () => request<string[]>("/api/v1/symbols"),
  latestFeatures: (symbol: string) =>
    request<LatestFeatures>(`/api/v1/latest-features?symbol=${encodeURIComponent(symbol)}`),
  predict: (symbol: string, features: FeatureVector) =>
    request<RiskPrediction>("/api/v1/predict-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, ...features }),
    }),
};
