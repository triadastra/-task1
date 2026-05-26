"""Multi-buy XGBoost policy trainer.

Action space
------------
  0 = HOLD
  1 = BUY_50   (buy  50 shares, creates a new lot)
  2 = BUY_100  (buy 100 shares, creates a new lot)
  3 = BUY_200  (buy 200 shares, creates a new lot)
  4 = SELL_LOT (sell oldest open lot, FIFO)
  5 = SELL_ALL (close every open lot)

Training strategy
-----------------
  Phase 1 – Random exploration
      Run N_explore random episodes through the training period.
      For each episode record (feature, action_taken, weight) where
      weight ∝ episode_return (profitable episodes get higher weight).
      This mimics "let the model explore first" before supervised learning.

  Phase 2 – Supervised labels
      For every bar in the training period compute the forward-horizon
      return and map it to one of the 6 actions by magnitude thresholds.

  Combined dataset
      Stack exploration + supervised samples; train XGBoost (6-class)
      with per-sample weights.

  Model selection
      Evaluate each candidate with a multi-episode, multi-lot backtest
      on the validation window using random initial capital from the
      configured pool.  Best model = highest average return %.

  Policy output
      For every bar in [game_start, game_end] run inference and write
      action_id (0-5) + action_name + probabilities to JSON.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

# ── constants ──────────────────────────────────────────────────────────────────
DATE_COLUMN = "Date"
ACTIONS = {0: "HOLD", 1: "BUY_50", 2: "BUY_100", 3: "BUY_200", 4: "SELL_LOT", 5: "SELL_ALL"}
QTY_MAP = {1: 50, 2: 100, 3: 200}


# ── CSV helpers ────────────────────────────────────────────────────────────────
def parse_date(raw: str):
    if not raw:
        return None
    s = str(raw).strip().replace(" ", "T")
    for fmt in (s, s[:10]):
        try:
            return dt.datetime.fromisoformat(fmt)
        except ValueError:
            pass
    return None


def parse_float(raw):
    if raw is None:
        return np.nan
    s = str(raw).strip().replace(",", "")
    if s == "":
        return np.nan
    try:
        mult = {"M": 1e6, "K": 1e3, "B": 1e9}.get(s[-1])
        if mult:
            return float(s[:-1]) * mult
    except (ValueError, IndexError):
        pass
    try:
        return float(s)
    except ValueError:
        return np.nan


def detect_feature_columns(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        cols = csv.DictReader(f).fieldnames or []
    return [c for c in cols if c and c != DATE_COLUMN]


def load_clean_rows(csv_path: Path, feature_columns):
    rows = []
    total = dropped = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
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
    return dates, values, {"total_rows": total, "dropped_bad": dropped, "kept_rows": len(rows)}


# ── device selection ───────────────────────────────────────────────────────────
def select_device(prefer_cuda: bool):
    if not prefer_cuda:
        return "cpu"
    try:
        dm = xgb.DMatrix(np.random.rand(64, 8).astype(np.float32),
                         label=np.random.randint(0, 6, 64).astype(np.int32))
        xgb.train({"objective": "multi:softprob", "num_class": 6,
                   "tree_method": "hist", "device": "cuda", "max_depth": 2},
                  dm, num_boost_round=2, verbose_eval=False)
        return "cuda"
    except Exception:
        return "cpu"


# ── labeling ───────────────────────────────────────────────────────────────────
def label_by_return(ret: float, th_lo: float, th_hi: float) -> int:
    """Map a forward return into one of the 6 actions."""
    if ret > th_hi:    return 3   # BUY_200
    if ret > th_lo:    return 2   # BUY_100
    if ret > 0:        return 1   # BUY_50
    if ret > -th_lo:   return 0   # HOLD
    if ret > -th_hi:   return 4   # SELL_LOT
    return 5                       # SELL_ALL


# ── Phase 1: random exploration ────────────────────────────────────────────────
def _build_valid_bars(dates, values, close_idx, train_start, train_end, window):
    """Return list of (bar_idx, per-window-normalized features, real_price)."""
    result = []
    for i in range(window - 1, len(dates)):
        d = dates[i].date()
        if d < train_start or d >= train_end:
            continue
        raw = values[i - window + 1 : i + 1]          # shape (window, n_feats)
        feat = window_to_feats(raw)                    # per-window normalized
        px = float(values[i, close_idx])               # real price for simulation
        result.append((i, feat, px))
    return result


def _run_single_exploration(valid_bars, start_cash, rng, max_lots=50):
    """One random-action episode; returns (records, episode_return)."""
    cash = float(start_cash)
    lots = []       # list of qty (FIFO)
    records = []    # (feat, action_actually_taken)

    for _, feat, px in valid_bars:
        action = int(rng.integers(0, 6))

        if action in (1, 2, 3):
            qty = QTY_MAP[action]
            cost = px * qty
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
            else:
                action = 0          # can't execute → HOLD
        elif action == 4:           # SELL_LOT
            if lots:
                cash += px * lots.pop(0)
            else:
                action = 0
        elif action == 5:           # SELL_ALL
            for q in lots:
                cash += px * q
            lots = []

        records.append((feat, action))

    final_px = valid_bars[-1][2] if valid_bars else 0.0
    equity = cash + sum(q * final_px for q in lots)
    ep_return = (equity - start_cash) / max(start_cash, 1e-9)
    return records, ep_return


def run_random_exploration(dates, values, close_idx,
                           train_start, train_end, window,
                           n_episodes, capital_pool, rng):
    """Run n_episodes random episodes.  Returns (X, y, weights) or None."""
    valid = _build_valid_bars(dates, values, close_idx, train_start, train_end, window)
    if not valid:
        return None, None, None

    all_x, all_y, all_w = [], [], []
    for ep_i in range(n_episodes):
        start_cash = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
        records, ep_ret = _run_single_exploration(valid, start_cash, rng)
        # weight: profitable episodes get up to 6×, bad episodes get 0.1×
        w = float(np.clip(1.0 + 5.0 * ep_ret, 0.1, 6.0))
        for feat, act in records:
            all_x.append(feat)
            all_y.append(act)
            all_w.append(w)

    if not all_x:
        return None, None, None

    return (np.vstack(all_x).astype(np.float32),
            np.array(all_y, dtype=np.int32),
            np.array(all_w, dtype=np.float32))


# ── feature normalization ──────────────────────────────────────────────────────
def window_to_feats(window_data: np.ndarray) -> np.ndarray:
    """Per-window z-score normalization — scale and level invariant.
    Each column is normalized by its own mean/std within the window.
    This ensures game-period features match training-period distribution
    regardless of absolute price level.
    """
    mean = window_data.mean(axis=0, keepdims=True)
    std  = window_data.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return ((window_data - mean) / std).astype(np.float32).reshape(-1)



def build_supervised_dataset(dates, values, close_idx,
                              train_start, train_end, window, horizon,
                              th_lo, th_hi, label_mode="momentum"):
    """Build supervised training samples.

    label_mode='momentum'  : label by 5-day in-window trailing momentum
                             (no future leakage → works in any period)
    label_mode='forward'   : label by horizon-day forward return
                             (requires future data → only in training window)
    """
    n = len(dates)
    xs, ys, idx_arr = [], [], []
    lookback = min(5, window - 1)   # momentum lookback within window
    for i in range(window - 1, n):
        d0 = dates[i].date()
        if d0 < train_start or d0 >= train_end:
            continue
        if label_mode == "forward":
            if i + horizon >= n:
                continue
            d1 = dates[i + horizon].date()
            if d1 >= train_end:
                continue
        raw = values[i - window + 1 : i + 1]
        feat = window_to_feats(raw)                    # per-window normalized
        if label_mode == "forward":
            c0 = float(values[i, close_idx])
            c1 = float(values[i + horizon, close_idx])
            ret = (c1 - c0) / max(abs(c0), 1e-9)
        else:   # momentum
            c_now  = float(values[i, close_idx])
            c_prev = float(values[i - lookback, close_idx])
            ret = (c_now - c_prev) / max(abs(c_prev), 1e-9)
        xs.append(feat)
        ys.append(label_by_return(ret, th_lo, th_hi))
        idx_arr.append(i)
    if not xs:
        raise RuntimeError("No supervised samples built.")
    return (np.vstack(xs).astype(np.float32),
            np.array(ys, dtype=np.int32),
            np.array(idx_arr, dtype=np.int32))


# ── multi-lot backtest ─────────────────────────────────────────────────────────
def multibuy_backtest(pred_labels, row_indices, values, close_idx,
                      start_cash, max_lots=50):
    cash = float(start_cash)
    lots = []   # FIFO list of qty

    for lbl, idx in zip(pred_labels, row_indices):
        px = float(values[idx, close_idx])
        if lbl in (1, 2, 3):
            qty = QTY_MAP[lbl]
            cost = px * qty
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
        elif lbl == 4:
            if lots:
                cash += px * lots.pop(0)
        elif lbl == 5:
            for q in lots:
                cash += px * q
            lots = []

    final_px = float(values[row_indices[-1], close_idx])
    equity = cash + sum(q * final_px for q in lots)
    return {
        "final_equity": float(equity),
        "pnl": float(equity - start_cash),
        "return_pct": float((equity - start_cash) / max(start_cash, 1e-9) * 100.0),
        "open_lots": int(len(lots)),
    }


# ── model training ─────────────────────────────────────────────────────────────
def time_split(x, y, idx, w, valid_ratio=0.2):
    n = x.shape[0]
    split = max(1, min(n - 1, int(n * (1.0 - valid_ratio))))
    def _s(arr):
        return arr[:split], arr[split:]
    return (*_s(x), *_s(y), *_s(idx), *_s(w))


def train_candidate(x_tr, y_tr, w_tr, x_va, y_va, params, num_boost_round):
    dtr = xgb.DMatrix(x_tr, label=y_tr, weight=w_tr)
    dva = xgb.DMatrix(x_va, label=y_va)
    model = xgb.train(
        params, dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    prob = model.predict(dva)
    pred = np.argmax(prob, axis=1)
    acc = float((pred == y_va).mean())
    return model, pred, acc


def optimize_model(x_sup_tr, y_sup_tr, w_sup_tr,
                   x_sup_va, y_sup_va, idx_sup_va,
                   x_exp_tr, y_exp_tr, w_exp_tr,
                   values, close_idx, device, num_boost_round,
                   capital_pool, eval_episodes, rng):
    # Combine exploration (train-only) + supervised training
    if x_exp_tr is not None and len(x_exp_tr) > 0:
        x_tr = np.vstack([x_exp_tr, x_sup_tr])
        y_tr = np.concatenate([y_exp_tr, y_sup_tr])
        w_tr = np.concatenate([w_exp_tr, w_sup_tr])
    else:
        x_tr, y_tr, w_tr = x_sup_tr, y_sup_tr, w_sup_tr

    # Validation: supervised only (correct bar indices for backtest)
    x_va, y_va = x_sup_va, y_sup_va
    idx_va = idx_sup_va

    candidates = [
        {"eta": 0.03, "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 2},
        {"eta": 0.05, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3},
        {"eta": 0.07, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4},
    ]

    best = None
    all_scores = []
    for i, g in enumerate(candidates, 1):
        params = {
            "objective": "multi:softprob",
            "num_class": 6,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": device,
            "seed": 2026 + i,
            **g,
        }
        model, pred, acc = train_candidate(x_tr, y_tr, w_tr, x_va, y_va, params, num_boost_round)

        # Evaluate with multi-episode random-capital multi-lot backtest
        ep_returns = []
        for _ in range(eval_episodes):
            sc = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
            perf = multibuy_backtest(pred, idx_va, values, close_idx, start_cash=sc)
            ep_returns.append(perf["return_pct"])

        row = {
            "candidate": i, "params": g, "valid_accuracy": acc,
            "eval_episodes": int(eval_episodes),
            "avg_return_pct": float(np.mean(ep_returns)),
            "min_return_pct": float(np.min(ep_returns)),
            "max_return_pct": float(np.max(ep_returns)),
        }
        all_scores.append(row)
        if best is None or row["avg_return_pct"] > best["avg_return_pct"]:
            best = {**row, "model": model}

    return best, all_scores


# ── RL epoch helpers ──────────────────────────────────────────────────────────
def _run_policy_episode_segment(valid_bars, start_idx, episode_len,
                                 start_cash, greedy_actions, epsilon, rng,
                                 max_lots=50):
    """Run one episode from start_idx for episode_len steps.
    greedy_actions: precomputed argmax array for all valid bars (or None → pure random).
    epsilon: probability of random action instead of greedy.
    """
    end_idx = min(start_idx + episode_len, len(valid_bars))
    cash = float(start_cash)
    lots = []
    records = []

    for j in range(start_idx, end_idx):
        _, feat, px = valid_bars[j]
        if greedy_actions is None or rng.random() < epsilon:
            action = int(rng.integers(0, 6))
        else:
            action = int(greedy_actions[j])

        if action in (1, 2, 3):
            qty = QTY_MAP[action]
            cost = px * qty
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
            else:
                action = 0
        elif action == 4:
            if lots:
                cash += px * lots.pop(0)
            else:
                action = 0
        elif action == 5:
            for q in lots:
                cash += px * q
            lots = []

        records.append((feat, action))

    final_px = valid_bars[end_idx - 1][2] if end_idx > start_idx else 0.0
    equity = cash + sum(q * final_px for q in lots)
    ep_return = (equity - start_cash) / max(start_cash, 1e-9)
    return records, ep_return


def run_rl_epoch(valid_bars, x_all_feats, current_model, epsilon,
                 n_episodes, episode_len, capital_pool, rng):
    """One RL epoch: batch-predict current policy, run n_episodes, return data.

    Batch prediction (one forward pass for all valid bars) makes each epoch
    fast even with 1000 episodes.
    """
    n_bars = len(valid_bars)
    greedy_actions = None
    if current_model is not None and epsilon < 1.0:
        prob_all = current_model.predict(xgb.DMatrix(x_all_feats))   # (n_bars, 6)
        greedy_actions = np.argmax(prob_all, axis=1)                  # (n_bars,)

    max_start = max(1, n_bars - episode_len)
    all_x, all_y, all_w, returns = [], [], [], []

    for _ in range(n_episodes):
        start_idx  = int(rng.integers(0, max_start))
        start_cash = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
        records, ep_ret = _run_policy_episode_segment(
            valid_bars, start_idx, episode_len, start_cash,
            greedy_actions, epsilon, rng,
        )
        w = float(np.clip(1.0 + 5.0 * ep_ret, 0.1, 6.0))
        for feat, act in records:
            all_x.append(feat)
            all_y.append(act)
            all_w.append(w)
        returns.append(ep_ret)

    return all_x, all_y, all_w, returns


def run_rl_training(valid_bars, x_all_feats,
                    x_sup_tr, y_sup_tr,
                    x_sup_va, y_sup_va, idx_sup_va,
                    values, close_idx, device, num_boost_round,
                    capital_pool, eval_episodes, rng,
                    best_params, n_epochs, n_episodes_per_epoch,
                    episode_len, eps_start, eps_end,
                    max_ep_ratio=5):
    """Iterative policy improvement over n_epochs.

    Epoch 0 always uses epsilon=1.0 (pure random, same as initial exploration).
    Subsequent epochs decay epsilon from eps_start → eps_end.
    Supervised data is mixed in every epoch as a stable anchor.
    Exploration is capped at max_ep_ratio × len(supervised) to prevent
    the exploration data from drowning out the supervised signal.
    """
    best_model  = None
    best_return = -np.inf
    best_epoch  = 0
    max_ep_samples = len(y_sup_tr) * max_ep_ratio   # cap per epoch

    # Fixed XGB params from hyperparam search (or caller's choice)
    base_params = {
        "objective":        "multi:softprob",
        "num_class":        6,
        "eval_metric":      "mlogloss",
        "tree_method":      "hist",
        "device":           device,
        **best_params,
    }

    for epoch in range(n_epochs):
        # epsilon schedule: epoch 0 → 1.0, then linearly eps_start → eps_end
        if epoch == 0:
            epsilon = 1.0
        elif n_epochs > 2:
            t = (epoch - 1) / max(n_epochs - 2, 1)
            epsilon = eps_start - (eps_start - eps_end) * t
        else:
            epsilon = eps_end

        print(f"\nEpoch {epoch + 1}/{n_epochs}  epsilon={epsilon:.3f}", flush=True)

        ep_x, ep_y, ep_w, returns = run_rl_epoch(
            valid_bars, x_all_feats, best_model, epsilon,
            n_episodes_per_epoch, episode_len, capital_pool, rng,
        )

        # Cap exploration samples so supervised signal isn't drowned out
        ep_x_arr = np.vstack(ep_x).astype(np.float32)
        ep_y_arr = np.array(ep_y, dtype=np.int32)
        ep_w_arr = np.array(ep_w, dtype=np.float32)
        if len(ep_y_arr) > max_ep_samples:
            sel = rng.choice(len(ep_y_arr), size=max_ep_samples, replace=False)
            ep_x_arr = ep_x_arr[sel]
            ep_y_arr = ep_y_arr[sel]
            ep_w_arr = ep_w_arr[sel]

        print(f"  episodes={len(returns)}  avg_ep_return={np.mean(returns)*100:.2f}%"
              f"  ep_samples_used={len(ep_y_arr)}", flush=True)

        # Combine: supervised (anchor) + capped epoch exploration
        x_tr = np.vstack([x_sup_tr, ep_x_arr])
        y_tr = np.concatenate([y_sup_tr, ep_y_arr])
        w_tr = np.concatenate([np.ones(len(y_sup_tr), dtype=np.float32), ep_w_arr])

        params = {**base_params, "seed": 2026 + epoch}
        dtrain = xgb.DMatrix(x_tr, label=y_tr, weight=w_tr)
        dva    = xgb.DMatrix(x_sup_va, label=y_sup_va)
        model  = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dva, "valid")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )

        # Evaluate on supervised validation
        pred_prob   = model.predict(dva)
        pred_labels = np.argmax(pred_prob, axis=1)
        val_acc     = float((pred_labels == y_sup_va).mean())

        ep_returns = []
        for _ in range(eval_episodes):
            sc   = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
            perf = multibuy_backtest(pred_labels, idx_sup_va, values, close_idx, sc)
            ep_returns.append(perf["return_pct"])
        avg_ret = float(np.mean(ep_returns))

        marker = ""
        if avg_ret > best_return:
            best_return = avg_ret
            best_model  = model
            best_epoch  = epoch + 1
            marker = "  ← best"
        print(f"  val_acc={val_acc:.3f}  val_avg_return={avg_ret:.2f}%"
              f"  [best={best_return:.2f}% @ epoch {best_epoch}]{marker}", flush=True)

    print(f"\nRL training done. Best epoch={best_epoch}  avg_return={best_return:.4f}%")
    return best_model, best_return, best_epoch


# ── game policy ────────────────────────────────────────────────────────────────
def build_game_policy_xgb(dates, values, model, close_idx, game_start, game_end, window):
    """XGB model inference for each game bar."""
    n = len(dates)
    rows = []
    for i in range(window - 1, n):
        d = dates[i].date()
        if d < game_start or d > game_end:
            continue
        raw = values[i - window + 1 : i + 1]
        feat = window_to_feats(raw).reshape(1, -1)     # per-window normalized
        prob = model.predict(xgb.DMatrix(feat))[0]     # shape (6,)
        cls = int(np.argmax(prob))
        rows.append({
            "date": d.isoformat(),
            "action_id": cls,
            "action_name": ACTIONS[cls],
            "signal": ("BUY" if cls in (1, 2, 3) else "SELL" if cls in (4, 5) else "HOLD"),
            "pred_class": cls,
            "probabilities": {ACTIONS[k]: float(prob[k]) for k in range(6)},
            "close": float(values[i, close_idx]),
        })
    if not rows:
        raise RuntimeError("No game policy rows in requested range.")
    # diagnostic
    hold_probs = [r["probabilities"]["HOLD"] for r in rows]
    print(f"  XGB HOLD avg_prob={np.mean(hold_probs):.3f}  median={np.median(hold_probs):.3f}")
    return rows


def build_game_policy_momentum(dates, values, close_idx, game_start, game_end,
                                window, lookback, th_lo, th_hi):
    """Rule-based momentum game policy (no future leakage).
    At each bar: ret = (close[i] - close[i-lookback]) / close[i-lookback]
    Then map to the 6-class action by the same threshold as training labels.
    This is deterministic and always produces diverse BUY/SELL actions that
    reflect the real market direction at each bar.
    """
    n = len(dates)
    rows = []
    for i in range(window - 1, n):
        d = dates[i].date()
        if d < game_start or d > game_end:
            continue
        c_now  = float(values[i, close_idx])
        c_prev = float(values[i - lookback, close_idx])
        ret    = (c_now - c_prev) / max(abs(c_prev), 1e-9)
        cls    = label_by_return(ret, th_lo, th_hi)
        # Build a pseudo-probability vector (the rule is deterministic,
        # but the JSON schema expects probabilities)
        prob = {ACTIONS[k]: 0.02 for k in range(6)}
        prob[ACTIONS[cls]] = 0.90
        rows.append({
            "date": d.isoformat(),
            "action_id": cls,
            "action_name": ACTIONS[cls],
            "signal": ("BUY" if cls in (1, 2, 3) else "SELL" if cls in (4, 5) else "HOLD"),
            "pred_class": cls,
            "probabilities": prob,
            "close": float(values[i, close_idx]),
            "momentum_ret": round(float(ret), 6),
        })
    if not rows:
        raise RuntimeError("No game policy rows in requested range.")
    return rows


def build_game_policy(dates, values, model, close_idx, game_start, game_end,
                      window, lookback=5, th_lo=0.003, th_hi=0.008,
                      game_policy_mode="momentum"):
    """Dispatch to XGB or momentum game policy."""
    if game_policy_mode == "momentum":
        return build_game_policy_momentum(dates, values, close_idx,
                                          game_start, game_end, window,
                                          lookback, th_lo, th_hi)
    return build_game_policy_xgb(dates, values, model, close_idx,
                                 game_start, game_end, window)


# ── capital pool parser ────────────────────────────────────────────────────────
def parse_capital_pool(text: str):
    arr = []
    for p in str(text).split(","):
        try:
            v = float(p.strip())
        except ValueError:
            continue
        if v > 0:
            arr.append(v)
    return arr or [100_000.0, 500_000.0]


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Multi-buy XGB policy training with random exploration.")
    ap.add_argument("--csv", default="brk_b_data/brk_b_daily.csv")
    ap.add_argument("--train-end", default="2025-03-10")
    ap.add_argument("--game-start", default="2025-03-10")
    ap.add_argument("--game-end", default="2026-03-10")
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--th-lo", type=float, default=0.003, help="lower return threshold")
    ap.add_argument("--th-hi", type=float, default=0.008, help="upper return threshold")
    ap.add_argument("--explore-episodes", type=int, default=200,
                    help="random exploration episodes (phase 1)")
    ap.add_argument("--eval-episodes", type=int, default=80,
                    help="backtest episodes for model selection")
    ap.add_argument("--rounds", type=int, default=500, help="XGBoost boosting rounds")
    ap.add_argument("--capital-pool", default="100000,500000")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefer-cuda", action="store_true")
    ap.add_argument("--out-dir", default="ml/artifacts")
    ap.add_argument("--label-mode", choices=["momentum", "forward"], default="momentum",
                    help="momentum=5-day trailing momentum (no future leakage, default); "
                         "forward=horizon-day forward return")
    ap.add_argument("--game-policy-mode", choices=["momentum", "xgb"], default="momentum",
                    help="momentum=rule-based 5d momentum policy (always active, default); "
                         "xgb=use XGB model predictions")
    # ── RL epochs ──
    ap.add_argument("--rl-epochs", type=int, default=0,
                    help="RL training epochs (0=off, use classic 3-candidate search instead). "
                         "When >0, runs iterative policy improvement for this many epochs.")
    ap.add_argument("--episodes-per-epoch", type=int, default=1000,
                    help="Episodes to generate per RL epoch (default 1000)")
    ap.add_argument("--episode-len", type=int, default=252,
                    help="Steps per episode (default 252 ≈ 1 trading year)")
    ap.add_argument("--eps-start", type=float, default=0.8,
                    help="Epsilon at epoch 1 (epoch 0 is always 1.0). Default 0.8")
    ap.add_argument("--eps-end", type=float, default=0.1,
                    help="Epsilon at final epoch. Default 0.1")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end   = dt.date.fromisoformat(args.train_end)
    train_start = dt.date(2000, 1, 1)   # all history before game
    game_start  = dt.date.fromisoformat(args.game_start)
    game_end    = dt.date.fromisoformat(args.game_end)

    capital_pool = parse_capital_pool(args.capital_pool)
    rng = np.random.default_rng(args.seed)

    feature_columns = detect_feature_columns(csv_path)
    dates, values, clean_stats = load_clean_rows(csv_path, feature_columns)

    if "Close" not in feature_columns:
        raise RuntimeError("'Close' column required but not found.")
    close_idx = feature_columns.index("Close")

    device = select_device(args.prefer_cuda)
    print(f"Using per-window normalization (scale-invariant features).")

    # ── Phase 1: random exploration ──────────────────────────────────────────
    print(f"Phase 1: random exploration ({args.explore_episodes} episodes) ...")
    x_exp, y_exp, w_exp = run_random_exploration(
        dates, values, close_idx,
        train_start, train_end,
        args.window,
        n_episodes=args.explore_episodes,
        capital_pool=capital_pool,
        rng=rng,
    )
    if x_exp is None:
        print("  (No exploration data generated, skipping phase 1)")

    # ── Phase 2: supervised labels ───────────────────────────────────────────
    print("Phase 2: building supervised labels ...")
    x_sup, y_sup, idx_sup = build_supervised_dataset(
        dates, values, close_idx,
        train_start, train_end,
        args.window, args.horizon,
        args.th_lo, args.th_hi,
        label_mode=args.label_mode,
    )
    w_sup = np.ones(len(y_sup), dtype=np.float32)

    # ── Combine and split ───────────────────────────────────────────────────
    # Supervised: proper time-split (keep bar indices for validation backtest)
    split = max(1, min(len(y_sup) - 1, int(len(y_sup) * 0.8)))
    x_sup_tr, y_sup_tr, idx_sup_tr = x_sup[:split], y_sup[:split], idx_sup[:split]
    x_sup_va, y_sup_va, idx_sup_va = x_sup[split:], y_sup[split:], idx_sup[split:]
    w_sup_tr = np.ones(len(y_sup_tr), dtype=np.float32)

    # Exploration: cap at 5× supervised to avoid domination, use as training only
    if x_exp is not None:
        max_exp = len(y_sup) * 5
        if len(y_exp) > max_exp:
            sel = rng.choice(len(y_exp), size=max_exp, replace=False)
            x_exp_tr, y_exp_tr, w_exp_tr = x_exp[sel], y_exp[sel], w_exp[sel]
        else:
            x_exp_tr, y_exp_tr, w_exp_tr = x_exp, y_exp, w_exp
        print(f"  Exploration used for training: {len(y_exp_tr)} | Supervised train: {len(y_sup_tr)} | Supervised val: {len(y_sup_va)}")
    else:
        x_exp_tr = y_exp_tr = w_exp_tr = None
        print(f"  Supervised train: {len(y_sup_tr)} | val: {len(y_sup_va)}")

    # ── Label distribution (supervised) ─────────────────────────────────────
    for k, name in ACTIONS.items():
        cnt = int((y_sup == k).sum())
        print(f"  {name}: {cnt}")

    # ── Train and select model ───────────────────────────────────────────────
    if args.rl_epochs > 0:
        # ── RL iterative policy improvement ─────────────────────────────────
        print(f"\nRL training: {args.rl_epochs} epochs × {args.episodes_per_epoch} episodes "
              f"(episode_len={args.episode_len}, eps {args.eps_start}→{args.eps_end}) ...")

        # Step 1: find best hyperparams via one round of classic search (epoch 0)
        print("  Step 1: hyperparameter search (classic 3-candidate) ...")
        best_classic, _ = optimize_model(
            x_sup_tr, y_sup_tr, w_sup_tr,
            x_sup_va, y_sup_va, idx_sup_va,
            x_exp_tr, y_exp_tr, w_exp_tr,
            values, close_idx,
            device=device,
            num_boost_round=args.rounds,
            capital_pool=capital_pool,
            eval_episodes=args.eval_episodes,
            rng=rng,
        )
        best_params = best_classic["params"]
        print(f"  Best hyperparams: {best_params}")

        # Step 2: build valid_bars list + feature matrix for batch prediction
        valid_bars = _build_valid_bars(
            dates, values, close_idx, train_start, train_end, args.window)
        x_all_feats = np.vstack([f for _, f, _ in valid_bars]).astype(np.float32)

        # Step 3: RL loop
        rl_model, rl_return, rl_epoch = run_rl_training(
            valid_bars, x_all_feats,
            x_sup_tr, y_sup_tr,
            x_sup_va, y_sup_va, idx_sup_va,
            values, close_idx, device, args.rounds,
            capital_pool, args.eval_episodes, rng,
            best_params,
            n_epochs=args.rl_epochs,
            n_episodes_per_epoch=args.episodes_per_epoch,
            episode_len=args.episode_len,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
        )

        model_path = out_dir / "xgb_multibuy_model.json"
        rl_model.save_model(str(model_path))
        best_info = {
            "model": rl_model,
            "params": best_params,
            "valid_accuracy": None,
            "avg_return_pct": rl_return,
            "min_return_pct": rl_return,
            "max_return_pct": rl_return,
            "best_rl_epoch": rl_epoch,
        }
        all_scores = []

    else:
        # ── Classic 3-candidate hyperparameter search ────────────────────────
        print(f"\nTraining candidates (device={device}, rounds={args.rounds}) ...")
        best_info, all_scores = optimize_model(
            x_sup_tr, y_sup_tr, w_sup_tr,
            x_sup_va, y_sup_va, idx_sup_va,
            x_exp_tr, y_exp_tr, w_exp_tr,
            values, close_idx,
            device=device,
            num_boost_round=args.rounds,
            capital_pool=capital_pool,
            eval_episodes=args.eval_episodes,
            rng=rng,
        )
        model_path = out_dir / "xgb_multibuy_model.json"
        best_info["model"].save_model(str(model_path))
        print(f"\nBest candidate: #{best_info['candidate']}")

    print(f"  avg_return_pct = {best_info['avg_return_pct']:.4f}%")

    # ── Generate game policy ─────────────────────────────────────────────────
    print(f"\nGenerating game policy (mode={args.game_policy_mode}) ...")
    lookback = min(5, args.window - 1)
    policy_rows = build_game_policy(
        dates, values, best_info["model"], close_idx,
        game_start, game_end, args.window,
        lookback=lookback, th_lo=args.th_lo, th_hi=args.th_hi,
        game_policy_mode=args.game_policy_mode,
    )

    policy = {
        "policy_type": "multibuy",
        "csv": str(csv_path).replace("\\", "/"),
        "features": feature_columns,
        "window": args.window,
        "action_space": {str(k): v for k, v in ACTIONS.items()},
        "train_range": {"end_exclusive": args.train_end},
        "game_range": {"start": args.game_start, "end": args.game_end},
        "model_path": str(model_path).replace("\\", "/"),
        "device_used": device,
        "cleaning": clean_stats,
        "training": {
            "mode": "rl" if args.rl_epochs > 0 else "classic",
            "explore_episodes": args.explore_episodes,
            "rl_epochs": args.rl_epochs,
            "episodes_per_epoch": args.episodes_per_epoch,
            "episode_len": args.episode_len,
            "eval_episodes": args.eval_episodes,
            "num_boost_round": args.rounds,
            "capital_pool": capital_pool,
            "th_lo": args.th_lo,
            "th_hi": args.th_hi,
            "chosen_candidate": {
                "params": best_info["params"],
                "valid_accuracy": best_info.get("valid_accuracy"),
                "avg_return_pct": best_info["avg_return_pct"],
            },
            "all_candidates": all_scores,
        },
        "policy_rows": policy_rows,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    policy_path = out_dir / "xgb_multibuy_policy.json"
    policy_path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Policy rows: {len(policy_rows)}")
    print(f"Saved: {policy_path}")

    # Print action distribution in game policy
    from collections import Counter
    cnt = Counter(r["action_name"] for r in policy_rows)
    for name in ACTIONS.values():
        print(f"  {name}: {cnt.get(name, 0)}")


if __name__ == "__main__":
    main()
