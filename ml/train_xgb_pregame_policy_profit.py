import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

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


def detect_feature_columns(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
    return [c for c in cols if c and c != DATE_COLUMN]


def load_clean_rows(csv_path: Path, feature_columns):
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

            vals = np.array([parse_float(r.get(c)) for c in feature_columns], dtype=np.float64)
            if not np.isfinite(vals).all():
                dropped += 1
                continue

            rows.append((d, vals.astype(np.float32)))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError("No valid rows after cleaning.")

    dates = [r[0] for r in rows]
    values = np.vstack([r[1] for r in rows]).astype(np.float32)
    stats = {"total_rows": total, "dropped_bad": dropped, "kept_rows": int(values.shape[0])}
    return dates, values, stats


def select_device(prefer_cuda: bool):
    if not prefer_cuda:
        return "cpu"
    try:
        x = np.random.rand(64, 16).astype(np.float32)
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


def build_signal_dataset(dates, values, close_idx, train_end_date: dt.date, window: int, horizon: int, threshold: float):
    n = len(dates)
    xs, ys, row_indices = [], [], []

    for i in range(window - 1, n - horizon):
        d_i = dates[i].date()
        d_f = dates[i + horizon].date()
        if d_i >= train_end_date or d_f >= train_end_date:
            continue

        feat = values[i - window + 1 : i + 1].reshape(-1)
        c_now = values[i, close_idx]
        c_fut = values[i + horizon, close_idx]
        ret = (c_fut - c_now) / max(abs(c_now), 1e-6)

        if ret > threshold:
            y = 2
        elif ret < -threshold:
            y = 0
        else:
            y = 1

        xs.append(feat)
        ys.append(y)
        row_indices.append(i)

    if not xs:
        raise RuntimeError("No training samples built.")

    return np.vstack(xs).astype(np.float32), np.array(ys, dtype=np.int32), np.array(row_indices, dtype=np.int32)


def backtest_by_pred_labels(pred_labels, row_indices, values, close_idx, start_cash=100000.0, lot_size=100):
    cash = float(start_cash)
    shares = 0

    for lbl, idx in zip(pred_labels, row_indices):
        px = float(values[idx, close_idx])
        if lbl == 2:
            cost = px * lot_size
            if cost <= cash:
                cash -= cost
                shares += lot_size
        elif lbl == 0:
            if shares >= lot_size:
                cash += px * lot_size
                shares -= lot_size

    last_px = float(values[row_indices[-1], close_idx])
    equity = cash + shares * last_px
    return {
        "final_equity": float(equity),
        "pnl": float(equity - start_cash),
        "return_pct": float((equity - start_cash) / start_cash * 100.0),
        "end_shares": int(shares),
    }


def time_split(x, y, idx, valid_ratio=0.2):
    n = x.shape[0]
    split = max(1, min(n - 1, int(n * (1.0 - valid_ratio))))
    return (x[:split], y[:split], idx[:split]), (x[split:], y[split:], idx[split:])


def parse_capital_pool(text: str):
    arr = []
    for p in str(text).split(","):
        try:
            v = float(p.strip())
        except ValueError:
            continue
        if v > 0:
            arr.append(v)
    return arr if arr else [100000.0, 500000.0]


def train_candidate(x_tr, y_tr, x_va, y_va, params, num_boost_round):
    dtr = xgb.DMatrix(x_tr, label=y_tr)
    dva = xgb.DMatrix(x_va, label=y_va)
    model = xgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=35,
        verbose_eval=False,
    )
    prob = model.predict(dva)
    pred = np.argmax(prob, axis=1)
    acc = float((pred == y_va).mean())
    return model, pred, acc


def optimize_model_for_profit(
    x,
    y,
    idx,
    values,
    close_idx,
    device,
    lot_size,
    capital_pool,
    episodes,
    random_seed,
    num_boost_round,
):
    (x_tr, y_tr, idx_tr), (x_va, y_va, idx_va) = time_split(x, y, idx, valid_ratio=0.2)

    candidate_grid = [
        {"eta": 0.03, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2},
        {"eta": 0.05, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"eta": 0.07, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4},
    ]

    best = None
    all_scores = []
    rng = np.random.default_rng(random_seed)

    for i, g in enumerate(candidate_grid, start=1):
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": device,
            "seed": 2026 + i,
            **g,
        }

        model, pred, acc = train_candidate(x_tr, y_tr, x_va, y_va, params, num_boost_round)
        episode_returns = []
        episode_pnls = []
        episode_equities = []
        for _ in range(episodes):
            start_cash = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
            perf = backtest_by_pred_labels(pred, idx_va, values, close_idx, start_cash=start_cash, lot_size=lot_size)
            episode_returns.append(perf["return_pct"])
            episode_pnls.append(perf["pnl"])
            episode_equities.append(perf["final_equity"])

        avg_return = float(np.mean(episode_returns))
        avg_pnl = float(np.mean(episode_pnls))
        avg_equity = float(np.mean(episode_equities))
        min_return = float(np.min(episode_returns))
        max_return = float(np.max(episode_returns))

        row = {
            "candidate": i,
            "params": g,
            "valid_accuracy": acc,
            "episodes": int(episodes),
            "avg_return_pct": avg_return,
            "avg_pnl": avg_pnl,
            "avg_final_equity": avg_equity,
            "min_return_pct": min_return,
            "max_return_pct": max_return,
        }
        all_scores.append(row)

        if best is None or row["avg_return_pct"] > best["avg_return_pct"]:
            best = {**row, "model": model}

    return best, all_scores


def build_game_policy(dates, values, model, close_idx, game_start: dt.date, game_end: dt.date, window: int):
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
                "close": float(values[i, close_idx]),
            }
        )

    if not rows:
        raise RuntimeError("No game policy rows in requested range.")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Profit-optimized pre-game XGB policy training.")
    parser.add_argument("--csv", default="brk_b_data/brk_b_daily.csv")
    parser.add_argument("--train-end", default="2025-03-10")
    parser.add_argument("--game-start", default="2025-03-10")
    parser.add_argument("--game-end", default="2026-03-10")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--capital-pool", default="100000,500000")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rounds", type=int, default=450)
    parser.add_argument("--prefer-cuda", action="store_true")
    parser.add_argument("--out-dir", default="ml/artifacts")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end = dt.date.fromisoformat(args.train_end)
    game_start = dt.date.fromisoformat(args.game_start)
    game_end = dt.date.fromisoformat(args.game_end)

    feature_columns = detect_feature_columns(csv_path)
    dates, values, clean_stats = load_clean_rows(csv_path, feature_columns)

    if "Close" not in feature_columns:
        raise RuntimeError("Close column is required for labeling/backtest but not found.")
    close_idx = feature_columns.index("Close")
    capital_pool = parse_capital_pool(args.capital_pool)

    device = select_device(args.prefer_cuda)

    x, y, row_idx = build_signal_dataset(
        dates,
        values,
        close_idx,
        train_end,
        window=args.window,
        horizon=args.horizon,
        threshold=args.threshold,
    )

    best, scores = optimize_model_for_profit(
        x,
        y,
        row_idx,
        values,
        close_idx,
        device=device,
        lot_size=args.lot_size,
        capital_pool=capital_pool,
        episodes=args.episodes,
        random_seed=args.seed,
        num_boost_round=args.rounds,
    )

    model_path = out_dir / "xgb_signal_pregame_model.json"
    best["model"].save_model(str(model_path))

    policy_rows = build_game_policy(
        dates,
        values,
        best["model"],
        close_idx,
        game_start=game_start,
        game_end=game_end,
        window=args.window,
    )

    policy = {
        "csv": str(csv_path).replace("\\", "/"),
        "features": feature_columns,
        "window": args.window,
        "train_range": {"end_exclusive": args.train_end},
        "game_range": {"start": args.game_start, "end": args.game_end},
        "label_map": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "model_path": str(model_path).replace("\\", "/"),
        "device_used": device,
        "cleaning": clean_stats,
        "profit_objective": {
            "capital_pool": capital_pool,
            "episodes": args.episodes,
            "lot_size": args.lot_size,
            "num_boost_round": args.rounds,
            "chosen_candidate": {
                "params": best["params"],
                "valid_accuracy": best["valid_accuracy"],
                "avg_return_pct": best["avg_return_pct"],
                "avg_pnl": best["avg_pnl"],
                "avg_final_equity": best["avg_final_equity"],
                "min_return_pct": best["min_return_pct"],
                "max_return_pct": best["max_return_pct"],
            },
            "all_candidates": scores,
        },
        "policy_rows": policy_rows,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    policy_path = out_dir / "xgb_policy_game.json"
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Profit-optimized pre-game policy generation completed.")
    print(f"Device: {device}")
    print(f"Features used: {len(feature_columns)} -> {feature_columns}")
    print(f"Capital pool: {capital_pool} | episodes={args.episodes}")
    print(f"Boost rounds: {args.rounds}")
    print(f"Best avg return (valid): {best['avg_return_pct']:.4f}%")
    print(f"Policy rows: {len(policy_rows)}")
    print(f"Saved: {policy_path}")


if __name__ == "__main__":
    main()
