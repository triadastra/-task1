import argparse
import csv
import datetime as dt
import json
import os
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

    total = 0
    dropped_bad = 0
    dropped_range = 0
    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            d = parse_date(r.get(DATE_COLUMN))
            if d is None:
                dropped_bad += 1
                continue

            day = d.date()
            if day < start or day > end:
                dropped_range += 1
                continue

            vals = [parse_float(r.get(c)) for c in FEATURE_COLUMNS]
            arr = np.array(vals, dtype=np.float64)
            if not np.isfinite(arr).all():
                dropped_bad += 1
                continue

            rows.append((d, arr))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError("No valid rows after cleaning and date filtering.")

    dates = [r[0] for r in rows]
    values = np.vstack([r[1] for r in rows]).astype(np.float32)

    stats = {
        "total_rows": total,
        "kept_rows": int(values.shape[0]),
        "dropped_bad": dropped_bad,
        "dropped_out_of_range": dropped_range,
    }
    return dates, values, stats


def select_device(prefer_cuda: bool):
    if not prefer_cuda:
        return "cpu"
    try:
        x = np.random.rand(64, 8).astype(np.float32)
        y = np.random.randint(0, 3, size=64).astype(np.int32)
        dm = xgb.DMatrix(x, label=y)
        xgb.train(
            {
                "objective": "multi:softprob",
                "num_class": 3,
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 2,
                "eta": 0.3,
            },
            dm,
            num_boost_round=2,
            verbose_eval=False,
        )
        return "cuda"
    except Exception:
        return "cpu"


def build_signal_dataset(values: np.ndarray, window: int, horizon: int, threshold: float):
    close_idx = FEATURE_COLUMNS.index("Close")
    n = values.shape[0]
    xs = []
    ys = []

    for i in range(window - 1, n - horizon):
        win = values[i - window + 1 : i + 1].reshape(-1)
        c_now = values[i, close_idx]
        c_fut = values[i + horizon, close_idx]
        ret = (c_fut - c_now) / max(abs(c_now), 1e-6)

        if ret > threshold:
            y = 2  # BUY
        elif ret < -threshold:
            y = 0  # SELL
        else:
            y = 1  # HOLD

        xs.append(win)
        ys.append(y)

    if not xs:
        raise RuntimeError("Signal dataset is empty. Adjust date range/window/horizon.")
    return np.vstack(xs).astype(np.float32), np.array(ys, dtype=np.int32)


def build_forecast_dataset(values: np.ndarray, window: int, steps: int):
    close_idx = FEATURE_COLUMNS.index("Close")
    n = values.shape[0]
    xs = []
    ys = []

    for i in range(window - 1, n - steps):
        win = values[i - window + 1 : i + 1].reshape(-1)
        future_close = values[i + 1 : i + 1 + steps, close_idx]
        xs.append(win)
        ys.append(future_close)

    if not xs:
        raise RuntimeError("Forecast dataset is empty. Adjust date range/window/steps.")
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32)


def time_split(x: np.ndarray, y: np.ndarray, valid_ratio: float):
    n = x.shape[0]
    split = max(1, min(n - 1, int(n * (1.0 - valid_ratio))))
    return x[:split], y[:split], x[split:], y[split:]


def train_signal_model(x, y, device: str):
    x_tr, y_tr, x_va, y_va = time_split(x, y, valid_ratio=0.2)
    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dva = xgb.DMatrix(x_va, label=y_va)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": device,
        "eta": 0.05,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtr,
        num_boost_round=500,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    prob = model.predict(dva)
    pred = np.argmax(prob, axis=1)
    acc = float((pred == y_va).mean())
    return model, {"valid_accuracy": acc, "valid_size": int(y_va.shape[0])}


def train_forecast_models(x, y_steps, device: str):
    x_tr, y_tr, x_va, y_va = time_split(x, y_steps, valid_ratio=0.2)
    dtr_base = xgb.DMatrix(x_tr)
    dva_base = xgb.DMatrix(x_va)

    models = []
    rmses = []
    for step in range(y_steps.shape[1]):
        dtr = xgb.DMatrix(x_tr, label=y_tr[:, step])
        dva = xgb.DMatrix(x_va, label=y_va[:, step])
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": device,
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "seed": 42 + step,
        }
        model = xgb.train(
            params,
            dtr,
            num_boost_round=400,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        pred = model.predict(dva)
        rmse = float(np.sqrt(np.mean((pred - y_va[:, step]) ** 2)))
        models.append(model)
        rmses.append(rmse)

    return models, {
        "valid_size": int(y_va.shape[0]),
        "rmse_by_step": rmses,
        "rmse_mean": float(np.mean(rmses)),
    }


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost signal + 47-step forecast models.")
    parser.add_argument("--csv", default="brk_b_data/brk_b_daily.csv", help="Input CSV path")
    parser.add_argument("--start-date", default="2025-03-10", help="Filter start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-03-10", help="Filter end date YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=14, help="Input window size")
    parser.add_argument("--forecast-steps", type=int, default=47, help="Forecast horizon steps")
    parser.add_argument("--signal-horizon", type=int, default=1, help="Signal label horizon in steps")
    parser.add_argument("--signal-threshold", type=float, default=0.002, help="Return threshold for BUY/SELL labels")
    parser.add_argument("--prefer-cuda", action="store_true", help="Try to train on CUDA first")
    parser.add_argument("--out-dir", default="ml/artifacts", help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    dates, values, clean_stats = load_clean_daily_rows(csv_path, args.start_date, args.end_date)
    device = select_device(args.prefer_cuda)

    x_sig, y_sig = build_signal_dataset(values, args.window, args.signal_horizon, args.signal_threshold)
    sig_model, sig_metrics = train_signal_model(x_sig, y_sig, device)

    x_for, y_for = build_forecast_dataset(values, args.window, args.forecast_steps)
    forecast_models, forecast_metrics = train_forecast_models(x_for, y_for, device)

    signal_model_path = out_dir / "xgb_signal_model.json"
    sig_model.save_model(str(signal_model_path))

    forecast_paths = []
    for i, m in enumerate(forecast_models, start=1):
        p = out_dir / f"xgb_forecast_step_{i:02d}.json"
        m.save_model(str(p))
        forecast_paths.append(str(p).replace("\\", "/"))

    meta = {
        "csv": str(csv_path).replace("\\", "/"),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "features": FEATURE_COLUMNS,
        "window": args.window,
        "forecast_steps": args.forecast_steps,
        "signal": {
            "horizon": args.signal_horizon,
            "threshold": args.signal_threshold,
            "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
            "model_path": str(signal_model_path).replace("\\", "/"),
            "metrics": sig_metrics,
        },
        "forecast": {
            "model_paths": forecast_paths,
            "metrics": forecast_metrics,
        },
        "cleaning": clean_stats,
        "device_used": device,
        "train_rows": int(values.shape[0]),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Training completed.")
    print(f"Device: {device}")
    print(f"Signal valid accuracy: {sig_metrics['valid_accuracy']:.4f}")
    print(f"Forecast mean RMSE: {forecast_metrics['rmse_mean']:.4f}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
