from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "ts",
    "asset",
    "power_avg",
    "cycle_time",
    "items",
    "status_time",
    "status",
    "alarm",
}


def run_inference(df_raw: pd.DataFrame, asset_id: int, models_dir: str | Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_raw.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid timestamps after parsing")

    for col in ["power_avg", "cycle_time", "items", "status_time", "status", "alarm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["alarm"] = df["alarm"].fillna(0).astype(int).clip(lower=0, upper=1)
    df["ts_diff"] = df["ts"].diff().dt.total_seconds()
    df["is_on"] = (df["power_avg"].fillna(0) > 0).astype(int)
    df["status"] = df["status"].ffill()
    df["session_break"] = (df["ts_diff"] > 300).fillna(False).astype(int)
    df["session_id"] = df["session_break"].cumsum().astype(int)

    models_path = Path(models_dir)
    clip_json_path = models_path / f"asset{asset_id}_clip_thresholds.json"
    if not clip_json_path.exists():
        raise FileNotFoundError(f"Missing clip thresholds file: {clip_json_path}")

    try:
        with clip_json_path.open("r", encoding="utf-8") as f:
            clip_thresholds = json.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse clip thresholds JSON: {exc}") from exc

    def clip_bounds(col_name: str) -> tuple[float | None, float | None]:
        value = clip_thresholds.get(col_name)
        if isinstance(value, dict):
            low  = value.get("low",  value.get("lower", value.get("min")))
            high = value.get("high", value.get("upper", value.get("max")))
            return low, high
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return value[0], value[1]
        return None, None

    for col in ["power_avg", "cycle_time", "items", "status_time"]:
        low, high = clip_bounds(col)
        if low is not None or high is not None:
            df[col] = df[col].clip(lower=low, upper=high)

    def rolling_sess(series: pd.Series, session_ids: pd.Series, window: int, func: str) -> pd.Series:
        min_periods = max(1, window // 4)

        def run_roll(s: pd.Series) -> pd.Series:
            roller = s.rolling(window=window, min_periods=min_periods)
            if func == "mean":
                return roller.mean()
            if func == "std":
                return roller.std(ddof=0)
            raise ValueError(f"Unsupported rolling function: {func}")

        return series.groupby(session_ids, sort=False).transform(run_roll)

    df["power_roll_mean_m"] = rolling_sess(df["power_avg"],  df["session_id"], 60,  "mean")
    df["power_roll_std_m"]  = rolling_sess(df["power_avg"],  df["session_id"], 60,  "std")
    df["power_diff_abs"]    = df["power_avg"].diff().abs().fillna(0.0)
    df["cycle_roll_mean_l"] = rolling_sess(df["cycle_time"], df["session_id"], 180, "mean")
    df["alarm_roll_m"]      = rolling_sess(df["alarm"].astype(float), df["session_id"], 60, "mean")

    counter = 0
    time_since_alarm = []
    for status in df["status"].tolist():
        if status == 3.0:
            counter = 0
        else:
            counter += 1
        time_since_alarm.append(counter)
    df["time_since_alarm"] = pd.Series(time_since_alarm, index=df.index, dtype=float)

    scaler_if_path  = models_path / f"asset{asset_id}_scaler_if.pkl"
    iso_path        = models_path / f"asset{asset_id}_iso_forest.pkl"
    scaler_xgb_path = models_path / f"asset{asset_id}_scaler_xgb.pkl"
    xgb_path        = models_path / f"asset{asset_id}_xgb_model.pkl"
    missing_files   = [str(p) for p in [scaler_if_path, iso_path, scaler_xgb_path, xgb_path] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing model files for asset {asset_id}: {missing_files}")

    scaler_if  = joblib.load(scaler_if_path)
    iso_forest = joblib.load(iso_path)
    scaler_xgb = joblib.load(scaler_xgb_path)
    xgb_model  = joblib.load(xgb_path)

    if_features = ["power_roll_mean_m", "power_roll_std_m", "power_diff_abs", "time_since_alarm"]
    df_if = df.dropna(subset=if_features).copy()
    if df_if.empty:
        raise ValueError("Not enough valid rows after dropping NaNs for IF features")

    x_if = scaler_if.transform(df_if[if_features])
    df_if["if_raw_score"] = iso_forest.decision_function(x_if)

    score_min = float(df_if["if_raw_score"].min())
    score_max = float(df_if["if_raw_score"].max())
    if np.isclose(score_min, score_max):
        df_if["health_raw"] = 50.0
    else:
        df_if["health_raw"] = ((df_if["if_raw_score"] - score_min) / (score_max - score_min)) * 100.0

    df_if["health_smooth"] = df_if["health_raw"].ewm(span=30, adjust=False).mean()
    df_if["health_band"]   = np.where(
        df_if["health_smooth"] >= 75, "healthy",
        np.where(df_if["health_smooth"] >= 50, "degrading", "alert"),
    )

    xgb_features = [
        "power_roll_mean_m", "power_roll_std_m", "power_diff_abs", "time_since_alarm",
        "cycle_roll_mean_l", "is_on", "alarm_roll_m", "health_smooth", "if_raw_score",
    ]
    df_xgb = df_if.dropna(subset=xgb_features).copy()
    if df_xgb.empty:
        raise ValueError("Not enough valid rows after dropping NaNs for XGB features")

    x_xgb     = scaler_xgb.transform(df_xgb[xgb_features])
    xgb_proba = np.asarray(xgb_model.predict_proba(x_xgb))
    if xgb_proba.ndim != 2 or xgb_proba.shape[1] < 2:
        raise ValueError("XGB model predict_proba did not return binary probabilities")

    df_xgb["xgb_pre_alarm_prob"] = xgb_proba[:, 1]
    df_xgb["combined_alert"] = (
        (df_xgb["health_band"] == "alert") & (df_xgb["xgb_pre_alarm_prob"] > 0.5)
    ).astype(int)

    return df_xgb[[
        "ts", "health_smooth", "health_band", "xgb_pre_alarm_prob",
        "combined_alert", "alarm", "is_on", "if_raw_score",
    ]].copy()


def run_cold_warm(
    df_raw: pd.DataFrame,
    borrow_asset_id: int,
    models_dir: str | Path,
    cold_rows: int = 50000,
    warm_rows: int = 50000,
) -> dict:
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb
    import warnings
    warnings.filterwarnings("ignore")

    RANDOM_STATE = 42
    IF_FEATURES  = ["power_roll_mean_m", "power_roll_std_m", "power_diff_abs", "time_since_alarm"]
    XGB_FEATURES = [
        "power_roll_mean_m", "power_roll_std_m", "power_diff_abs", "time_since_alarm",
        "cycle_roll_mean_l", "is_on", "alarm_roll_m", "health_smooth", "if_raw_score",
    ]
    LOOKAHEAD = 30

    def _featurize(df: pd.DataFrame, clip_thresh: dict | None = None):
        df = df.copy().reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        for col in ["power_avg", "cycle_time", "items", "status_time", "status", "alarm"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["alarm"]         = df["alarm"].fillna(0).astype(int).clip(0, 1)
        df["ts_diff"]       = df["ts"].diff().dt.total_seconds()
        df["is_on"]         = (df["power_avg"].fillna(0) > 0).astype(int)
        df["status"]        = df["status"].ffill()
        df["session_break"] = (df["ts_diff"] > 300).fillna(False).astype(int)
        df["session_id"]    = df["session_break"].cumsum().astype(int)

        clip_cols = ["power_avg", "cycle_time", "items", "status_time"]
        if clip_thresh is None:
            clip_thresh = {}
            for feat in clip_cols:
                p25, p75, p99 = df[feat].quantile([0.25, 0.75, 0.99])
                iqr   = p75 - p25
                upper = min(p75 + 3.0 * iqr, p99)
                clip_thresh[feat] = {"lower": 0.0, "upper": round(float(upper), 4)}
        for feat, bounds in clip_thresh.items():
            if feat in df.columns:
                low  = bounds.get("lower", bounds.get("low"))
                high = bounds.get("upper", bounds.get("high"))
                df[feat] = df[feat].clip(lower=low, upper=high)

        def rolling_sess(series, session_ids, window, func):
            min_p = max(1, window // 4)
            def roll(s):
                r = s.rolling(window, min_periods=min_p)
                return r.mean() if func == "mean" else r.std(ddof=0)
            return series.groupby(session_ids, sort=False).transform(roll)

        df["power_roll_mean_m"] = rolling_sess(df["power_avg"],  df["session_id"], 60,  "mean")
        df["power_roll_std_m"]  = rolling_sess(df["power_avg"],  df["session_id"], 60,  "std")
        df["power_diff_abs"]    = df["power_avg"].diff().abs().fillna(0.0)
        df["cycle_roll_mean_l"] = rolling_sess(df["cycle_time"], df["session_id"], 180, "mean")
        df["alarm_roll_m"]      = rolling_sess(df["alarm"].astype(float), df["session_id"], 60, "mean")

        counter, tsa = 0, []
        for s in df["status"].tolist():
            counter = 0 if s == 3.0 else counter + 1
            tsa.append(counter)
        df["time_since_alarm"] = pd.Series(tsa, index=df.index, dtype=float)

        s3 = (df["status"] == 3.0).astype(float)
        df["pre_alarm"] = (
            s3.iloc[::-1].rolling(LOOKAHEAD, min_periods=1).max()
            .iloc[::-1].shift(-1).fillna(0).astype(int)
        )
        return df.dropna(subset=IF_FEATURES).reset_index(drop=True), clip_thresh

    def _score(df, scaler_if, iso, scaler_xgb, xgb_m):
        df   = df.copy()
        x_if = scaler_if.transform(df[IF_FEATURES])
        scores = iso.decision_function(x_if)
        df["if_raw_score"] = scores
        mn, mx = scores.min(), scores.max()
        df["health_raw"]    = 50.0 if np.isclose(mn, mx) else ((scores - mn) / (mx - mn)) * 100
        df["health_smooth"] = df["health_raw"].ewm(span=30, adjust=False).mean()
        df["health_band"]   = np.where(df["health_smooth"] >= 75, "healthy",
                              np.where(df["health_smooth"] >= 50, "degrading", "alert"))
        df_x  = df.dropna(subset=XGB_FEATURES).copy()
        x_xgb = scaler_xgb.transform(df_x[XGB_FEATURES])
        proba = np.asarray(xgb_m.predict_proba(x_xgb))
        df["xgb_pre_alarm_prob"] = np.nan
        df.loc[df_x.index, "xgb_pre_alarm_prob"] = proba[:, 1]
        df["combined_alert"] = (
            (df["health_band"] == "alert") & (df["xgb_pre_alarm_prob"] > 0.5)
        ).astype(int)
        return df

    def _metrics(df):
        from sklearn.metrics import roc_auc_score
        out = {}
        try:
            out["auc_if"] = round(roc_auc_score(df["alarm"], -df["if_raw_score"]), 4)
        except Exception:
            out["auc_if"] = None
        try:
            v = df["xgb_pre_alarm_prob"].notna()
            out["auc_xgb"] = round(roc_auc_score(df.loc[v, "alarm"], df.loc[v, "xgb_pre_alarm_prob"]), 4)
        except Exception:
            out["auc_xgb"] = None
        a = df[df["alarm"] == 1]["health_smooth"]
        n = df[df["alarm"] == 0]["health_smooth"]
        out["health_alarm"]  = round(float(a.mean()), 1) if len(a) else None
        out["health_normal"] = round(float(n.mean()), 1) if len(n) else None
        out["health_sep"]    = round(abs((out["health_alarm"] or 0) - (out["health_normal"] or 0)), 1)
        alarm_mask = df["alarm"] == 1
        starts     = df[alarm_mask & ~alarm_mask.shift(1, fill_value=False)].index.tolist()
        lead_times = []
        for idx in starts:
            pos    = df.index.get_loc(idx)
            window = df.iloc[max(0, pos - 60):pos + 1]
            early  = window[window["xgb_pre_alarm_prob"] > 0.5]
            lead_times.append(pos - df.index.get_loc(early.index[0]) if len(early) else 0)
        lt = pd.Series(lead_times)
        out["alarm_events"] = len(starts)
        out["lead_pct_any"] = round(float((lt > 0).mean()), 3) if len(lt) else 0
        out["lead_median"]  = int(lt.median()) if len(lt) else 0
        out["rows_scored"]  = len(df)
        # Sanitise any nan/inf floats before returning
        return {
            k: (None if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')) else v)
            for k, v in out.items()
        }

    def _to_chart(df, max_rows=2000):
        step = max(1, len(df) // max_rows)
        sub  = df.iloc[::step].copy()
        sub["ts"] = pd.to_datetime(sub["ts"]).dt.strftime("%Y-%m-%d %H:%M")
        cols = ["ts", "health_smooth", "xgb_pre_alarm_prob", "combined_alert", "alarm", "if_raw_score"]
        sub  = sub[[c for c in cols if c in sub.columns]]
        # Convert to dict then replace any nan/inf with None
        records = sub.to_dict(orient="records")
        clean = []
        for row in records:
            clean.append({
                k: (None if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')) else v)
                for k, v in row.items()
            })
        return clean

    # ── Split data ──────────────────────────────────────────────────────────
    df_raw    = df_raw.copy()
    total     = len(df_raw)
    cold_rows = min(cold_rows, total // 2)
    warm_rows = min(warm_rows, total - cold_rows)

    df_cold_raw = df_raw.iloc[:cold_rows].copy()
    df_warm_raw = df_raw.iloc[cold_rows:cold_rows + warm_rows].copy()

    # ── Load borrowed models ────────────────────────────────────────────────
    models_path = Path(models_dir)
    needed  = [f"asset{borrow_asset_id}_{s}.pkl" for s in ["scaler_if", "iso_forest", "scaler_xgb", "xgb_model"]]
    missing = [str(models_path / f) for f in needed if not (models_path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    si    = joblib.load(models_path / f"asset{borrow_asset_id}_scaler_if.pkl")
    iso   = joblib.load(models_path / f"asset{borrow_asset_id}_iso_forest.pkl")
    sx    = joblib.load(models_path / f"asset{borrow_asset_id}_scaler_xgb.pkl")
    xgb_m = joblib.load(models_path / f"asset{borrow_asset_id}_xgb_model.pkl")

    # ── Phase 1: cold (borrowed models) ────────────────────────────────────
    df_cold, clip_thresh = _featurize(df_cold_raw)
    df_cold_scored = _score(df_cold, si, iso, sx, xgb_m)
    cold_metrics   = _metrics(df_cold_scored)
    cold_chart     = _to_chart(df_cold_scored)

    # ── Train on cold data ──────────────────────────────────────────────────
    train_info   = {}
    warm_metrics = None
    warm_chart   = []
    try:
        df_normal = df_cold[
            (df_cold["status"].isin([1.0, 2.0])) &
            (df_cold["alarm"]     == 0) &
            (df_cold["pre_alarm"] == 0) &
            (df_cold["is_on"]     == 1)
        ].reset_index(drop=True)


        if len(df_normal) < 100:
            raise ValueError(f"Only {len(df_normal)} normal rows — not enough to train")

        df_running    = df_cold[df_cold["is_on"] == 1]
        contamination = float(np.clip((
            (df_running["status"]    == 3.0) |
            (df_running["alarm"]     == 1)   |
            (df_running["pre_alarm"] == 1)
        ).mean(), 0.01, 0.49))

        si2    = RobustScaler()
        X_norm = si2.fit_transform(df_normal[IF_FEATURES])
        iso2   = IsolationForest(
            n_estimators=300, contamination=contamination,
            max_samples="auto", random_state=RANDOM_STATE, n_jobs=-1,
        )
        iso2.fit(X_norm)

        df_cold2   = df_cold.dropna(subset=IF_FEATURES).reset_index(drop=True)
        if_scores2 = iso2.decision_function(si2.transform(df_cold2[IF_FEATURES]))
        mn, mx = if_scores2.min(), if_scores2.max()
        df_cold2["if_raw_score"]  = if_scores2
        df_cold2["health_raw"]    = ((if_scores2 - mn) / (mx - mn + 1e-9)) * 100
        df_cold2["health_smooth"] = df_cold2["health_raw"].ewm(span=30, adjust=False).mean()

        df_xgb   = df_cold2[XGB_FEATURES + ["pre_alarm"]].dropna().reset_index(drop=True)
        X        = df_xgb[XGB_FEATURES]
        y        = df_xgb["pre_alarm"]
        sx2      = RobustScaler()
        X_scaled = sx2.fit_transform(X)
        splits   = list(TimeSeriesSplit(n_splits=5).split(X_scaled))
        tr, te   = splits[-1]
        n_pos, n_neg = (y.iloc[tr] == 1).sum(), (y.iloc[tr] == 0).sum()
        spw  = n_neg / n_pos if n_pos > 0 else 1.0
        xgb2 = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="aucpr", early_stopping_rounds=30,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        xgb2.fit(X_scaled[tr], y.iloc[tr], eval_set=[(X_scaled[te], y.iloc[te])], verbose=False)

        train_info = {"contamination": round(contamination, 4), "normal_rows": len(df_normal)}

        # ── Phase 2: warm (self-trained models) ────────────────────────────
        df_warm, _     = _featurize(df_warm_raw, clip_thresh)
        df_warm_scored = _score(df_warm, si2, iso2, sx2, xgb2)
        warm_metrics   = _metrics(df_warm_scored)
        warm_chart     = _to_chart(df_warm_scored)

    except Exception as e:
        train_info["error"] = str(e)

    return {
        "total_rows":   total,
        "cold_rows":    cold_rows,
        "warm_rows":    warm_rows,
        "borrow_asset": borrow_asset_id,
        "train_info":   train_info,
        "cold_metrics": cold_metrics,
        "warm_metrics": warm_metrics,
        "cold_chart":   cold_chart,
        "warm_chart":   warm_chart,
    }
