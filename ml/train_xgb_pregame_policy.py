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


def load_clean_rows(csv_path: Path):
    rows = []
    total = 0
    dropped = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            d = parse_date(r.get(DATE_COLUMN))
            if d is None:
                dropped += 1
                continue
            vals = np.array([parse_float(r.get(c)) for c in FEATURE_COLUMNS], dtype=np.float64)
            if not np.isfinite(vals).all():
                dropped += 1
                continue
            rows.append((d, vals.astype(np.float32)))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError("No valid rows in CSV after cleaning.")

    dates = [r[0] for r in rows]
    values = np.vstack([r[1] for r in rows]).astype(np.float32)
    return dates, values, {"total_rows": total, "dropped_bad": dropped, "kept_rows": int(values.shape[0])}


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


def build_train_dataset(dates, values, train_end_date: dt.date, window: int, horizon: int, threshold: float):
    close_idx = FEATURE_COLUMNS.index("Close")
    n = len(dates)
    x, y = [], []

    for i in range(window - 1, n - horizon):
        d_i = dates[i].date()
        d_fut = dates[i + horizon].date()
        if d_i >= train_end_date:
            continue
        if d_fut >= train_end_date:
            continue

        feat = values[i - window + 1 : i + 1].reshape(-1)
        c_now = values[i, close_idx]
        c_fut = values[i + horizon, close_idx]
        ret = (c_fut - c_now) / max(abs(c_now), 1e-6)

        if ret > threshold:
            label = 2  # BUY
        elif ret < -threshold:
            label = 0  # SELL
        else:
            label = 1  # HOLD

        x.append(feat)
        y.append(label)

    if not x:
        raise RuntimeError("No train samples built. Check train_end_date/window/horizon.")
    return np.vstack(x).astype(np.float32), np.array(y, dtype=np.int32)


def train_signal_model(x, y, device: str):
    n = x.shape[0]
    split = max(1, min(n - 1, int(n * 0.8)))
    x_tr, y_tr = x[:split], y[:split]
    x_va, y_va = x[split:], y[split:]

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
        "seed": 2026,
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


def build_game_policy(dates, values, model, game_start: dt.date, game_end: dt.date, window: int):
    rows = []
    n = len(dates)

    for i in range(window - 1, n):
        d = dates[i].date()
        if d < game_start or d > game_end:
            continue

        feat = values[i - window + 1 : i + 1].reshape(1, -1).astype(np.float32)
        prob = model.predict(xgb.DMatrix(feat))[0]
        cls = int(np.argmax(prob))
        label = {0: "SELL", 1: "HOLD", 2: "BUY"}[cls]

        rows.append(
            {
                "date": d.isoformat(),
                "signal": label,
                "pred_class": cls,
                "probabilities": {
                    "SELL": float(prob[0]),
                    "HOLD": float(prob[1]),
                    "BUY": float(prob[2]),
                },
            }
        )

    if not rows:
        raise RuntimeError("No game policy rows produced for the requested game range.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train XGB on pre-game data and generate game-period auto-play policy.")
    parser.add_argument("--csv", default="brk_b_data/brk_b_daily.csv")
    parser.add_argument("--train-end", default="2025-03-10", help="Train on data strictly before this date")
    parser.add_argument("--game-start", default="2025-03-10")
    parser.add_argument("--game-end", default="2026-03-10")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--out-dir", default="ml/artifacts")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end = dt.date.fromisoformat(args.train_end)
    game_start = dt.date.fromisoformat(args.game_start)
    game_end = dt.date.fromisoformat(args.game_end)

    dates, values, clean_stats = load_clean_rows(csv_path)
    device = select_device(args.prefer_cuda)

    x_train, y_train = build_train_dataset(dates, values, train_end, args.window, args.horizon, args.threshold)
    model, metrics = train_signal_model(x_train, y_train, device)

    model_path = out_dir / "xgb_signal_pregame_model.json"
    model.save_model(str(model_path))

    policy_rows = build_game_policy(dates, values, model, game_start, game_end, args.window)

    policy = {
        "csv": str(csv_path).replace("\\", "/"),
        "features": FEATURE_COLUMNS,
        "window": args.window,
        "train_range": {"end_exclusive": args.train_end},
        "game_range": {"start": args.game_start, "end": args.game_end},
        "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "model_path": str(model_path).replace("\\", "/"),
        "device_used": device,
        "cleaning": clean_stats,
        "metrics": metrics,
        "policy_rows": policy_rows,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    policy_path = out_dir / "xgb_policy_game.json"
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Pre-game training and policy generation completed.")
    print(f"Device: {device}")
    print(f"Train valid accuracy: {metrics['valid_accuracy']:.4f}")
    print(f"Policy rows: {len(policy_rows)}")
    print(f"Saved: {policy_path}")


if __name__ == "__main__":
    main()
