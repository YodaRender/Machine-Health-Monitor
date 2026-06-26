"""
Machine Health Monitor — web API for CSV upload + inference.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sklearn.metrics import roc_auc_score

from pipeline import REQUIRED_COLUMNS, run_inference, run_cold_warm

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Machine Health Monitor")


def _list_asset_ids() -> list[int]:
    ids: list[int] = []
    for p in MODELS_DIR.glob("asset*_clip_thresholds.json"):
        m = re.search(r"asset(\d+)_clip", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def _format_report(df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("─" * 55)
    lines.append("  INFERENCE RESULTS")
    lines.append("─" * 55)
    lines.append(f"  Rows scored       : {len(df):,}")
    lines.append(f"  Alarm rate        : {df['alarm'].mean():.4f}")

    try:
        auc_if = roc_auc_score(df["alarm"], -df["if_raw_score"])
        lines.append(f"  IF  AUC           : {auc_if:.4f}")
    except Exception:
        lines.append("  IF  AUC           : n/a")

    try:
        valid = df["xgb_pre_alarm_prob"].notna()
        auc_xgb = roc_auc_score(
            df.loc[valid, "alarm"], df.loc[valid, "xgb_pre_alarm_prob"]
        )
        lines.append(f"  XGB AUC           : {auc_xgb:.4f}")
    except Exception:
        lines.append("  XGB AUC           : n/a")

    a = df[df["alarm"] == 1]
    n = df[df["alarm"] == 0]
    h_alarm  = a["health_smooth"].mean() if len(a) else float("nan")
    h_normal = n["health_smooth"].mean() if len(n) else float("nan")
    lines.append(f"  Health — alarm    : {h_alarm:.1f}")
    lines.append(f"  Health — normal   : {h_normal:.1f}")
    if pd.notna(h_alarm) and pd.notna(h_normal):
        lines.append(f"  Health separation : {abs(h_alarm - h_normal):.1f} pts")

    bands = df["health_band"].value_counts().to_dict()
    lines.append(f"  Health bands      : {bands}")

    alarm_mask   = df["alarm"] == 1
    alarm_starts = df[
        alarm_mask & (~alarm_mask.shift(1, fill_value=False))
    ].index.tolist()
    lead_times: list[int] = []
    for idx in alarm_starts:
        pos    = df.index.get_loc(idx)
        start  = max(0, pos - 60)
        window = df.iloc[start:pos + 1]
        early  = window[window["xgb_pre_alarm_prob"] > 0.5]
        if len(early) > 0:
            lead_times.append(pos - df.index.get_loc(early.index[0]))
        else:
            lead_times.append(0)

    lt = pd.Series(lead_times)
    if len(lt) > 0:
        lines.append(f"  Alarm events      : {len(alarm_starts)}")
        lines.append(f"  Lead warning >0   : {(lt > 0).sum()} ({(lt > 0).mean():.1%})")
        lines.append(f"  Lead warning >5   : {(lt > 5).sum()} ({(lt > 5).mean():.1%})")
        lines.append(f"  Median lead (rows): {int(lt.median())}")
    else:
        lines.append("  Alarm events      : 0")

    lines.append("─" * 55)
    return "\n".join(lines)


@app.get("/assets")
def get_assets():
    return {"assets": _list_asset_ids()}


@app.post("/predict")
async def predict(asset_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = REQUIRED_COLUMNS.difference(df_raw.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}. "
                   f"Expected: {sorted(REQUIRED_COLUMNS)}",
        )

    try:
        aid = int(asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid asset_id") from exc

    try:
        out = run_inference(df_raw, aid, MODELS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = _format_report(out)

    display = out.copy()
    display["ts"] = pd.to_datetime(display["ts"], utc=True).map(
        lambda t: t.isoformat() if pd.notna(t) else ""
    )
    display["health_smooth"]      = display["health_smooth"].round(4)
    display["xgb_pre_alarm_prob"] = display["xgb_pre_alarm_prob"].round(6)
    display["if_raw_score"]       = display["if_raw_score"].round(6)

    rows = display.to_dict(orient="records")
    return {"summary": summary, "rows": rows, "row_count": len(rows)}


@app.post("/warmstart")
async def warmstart(
    file: UploadFile = File(...),
    borrow_asset_id: str = Form(...),
    cold_rows: int = Form(50000),
    warm_rows: int = Form(50000),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = REQUIRED_COLUMNS.difference(df_raw.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        bid = int(borrow_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid borrow_asset_id") from exc

    try:
        result = run_cold_warm(df_raw, bid, MODELS_DIR, cold_rows, warm_rows)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import math
    def sanitise(obj):
        if isinstance(obj, dict):
            return {k: sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitise(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    return sanitise(result)



@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)
