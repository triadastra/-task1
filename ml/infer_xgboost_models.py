import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]
DATE_COLUMN = "Date"


def parse_date(raw: str):
    if not raw:
        return None
    s = str(raw).strip().replace(" ", "T")
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        try:
            return dt.datetime.fromisoformat(s[:10])
        except ValueError:
            return None


def parse_float(raw):
    if raw is None:
        return np.nan
    s = str(raw).strip().replace(",", "")
    if s == "":
        return np.nan
    if s.endswith("M"):
        try:
            return float(s[:-1]) * 1_000_000.0
        except ValueError:
            return np.nan
    if s.endswith("K"):
        try:
            return float(s[:-1]) * 1_000.0
        except ValueError:
            return np.nan
    if s.endswith("B"):
        try:
            return float(s[:-1]) * 1_000_000_000.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_clean_daily_rows(csv_path: Path, start_date: str, end_date: str):
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = parse_date(r.get(DATE_COLUMN))
            if d is None:
                continue
            day = d.date()
            if day < start or day > end:
                continue
            vals = [parse_float(r.get(c)) for c in FEATURE_COLUMNS]
            arr = np.array(vals, dtype=np.float64)
            if not np.isfinite(arr).all():
                continue
            rows.append((d, arr))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError("No valid rows for inference.")

    dates = [r[0] for r in rows]
    values = np.vstack([r[1] for r in rows]).astype(np.float32)
    return dates, values


def build_latest_window(values: np.ndarray, window: int):
    if values.shape[0] < window:
        raise RuntimeError(f"Not enough rows for window={window}. Have {values.shape[0]}")
    return values[-window:].reshape(1, -1).astype(np.float32)


def infer_signal(meta: dict, x_latest: np.ndarray):
    model = xgb.Booster()
    model.load_model(meta["signal"]["model_path"])
    prob = model.predict(xgb.DMatrix(x_latest))[0]
    pred_idx = int(np.argmax(prob))
    label_map = meta["signal"]["label_map"]
    label = label_map[str(pred_idx)]
    return {
        "pred_class": pred_idx,
        "pred_label": label,
        "probabilities": {
            label_map["0"]: float(prob[0]),
            label_map["1"]: float(prob[1]),
            label_map["2"]: float(prob[2]),
        },
    }


def infer_forecast(meta: dict, x_latest: np.ndarray):
    preds = []
    for i, p in enumerate(meta["forecast"]["model_paths"], start=1):
        model = xgb.Booster()
        model.load_model(p)
        y = float(model.predict(xgb.DMatrix(x_latest))[0])
        preds.append({"step": i, "pred_close": y})
    return preds


def main():
    parser = argparse.ArgumentParser(description="Run inference for signal and 47-step forecast models.")
    parser.add_argument("--meta", default="ml/artifacts/meta.json", help="Path to training meta.json")
    parser.add_argument("--csv", default=None, help="Optional CSV override")
    parser.add_argument("--start-date", default=None, help="Optional start date override")
    parser.add_argument("--end-date", default=None, help="Optional end date override")
    parser.add_argument("--out", default="ml/artifacts/inference_latest.json", help="Output JSON path")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    csv_path = Path(args.csv or meta["csv"])
    start_date = args.start_date or meta["date_range"]["start"]
    end_date = args.end_date or meta["date_range"]["end"]
    window = int(meta["window"])

    dates, values = load_clean_daily_rows(csv_path, start_date, end_date)
    x_latest = build_latest_window(values, window)

    signal_out = infer_signal(meta, x_latest)
    forecast_out = infer_forecast(meta, x_latest)

    result = {
        "csv": str(csv_path).replace("\\", "/"),
        "window": window,
        "latest_date": dates[-1].date().isoformat(),
        "rows_used": int(values.shape[0]),
        "features": meta["features"],
        "signal": signal_out,
        "forecast_next_47_steps": forecast_out,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Inference completed.")
    print(f"Latest date: {result['latest_date']}")
    print(f"Signal: {signal_out['pred_label']} | probs={signal_out['probabilities']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
