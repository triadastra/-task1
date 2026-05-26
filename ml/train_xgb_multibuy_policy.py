"""Multi-buy XGBoost policy trainer with portfolio-aware features.

Action space (3-class)
----------------------
  0 = HOLD
  1 = BUY   – buy X shares;  X = round(P(BUY)  × MAX_BUY_SHARES), capped by cash
  2 = SELL  – sell X shares; X = round(P(SELL) × total_held), FIFO partial exit

Portfolio features (5 dims appended to 121 market dims → 126 total)
--------------------------------------------------------------------
  cash_ratio     : cash / total_equity
  invested_ratio : market_value / total_equity
  position_size  : total_shares / (MAX_BUY_SHARES × MAX_LOTS)
  equity_growth  : tanh(equity / initial_cash − 1)
  cost_vs_price  : avg_cost / current_price − 1  (negative = in profit)

Training strategy
-----------------
  Phase 1 – Random exploration with portfolio simulation
      Run N_explore random episodes; record (market_feat ++ portfolio_feat,
      oracle_label) per bar.  Oracle label = label_by_return(ret) — NOT the
      random action taken.  Diverse states + clean targets.

  Phase 2 – Supervised labels (neutral portfolio state)
      For every training bar: market_feat ++ neutral_portfolio_feat.
      Neutral = all cash, no positions.  Label via label_by_return.

  Combined dataset
      Exploration (126-dim, live portfolio) + Supervised (126-dim, neutral).
      Train XGBoost 3-class multi:softprob with per-sample weights.

  RL iterative improvement
      Each epoch: random OR greedy episodes (portfolio-aware).
      Greedy: sequential per-bar inference with live portfolio features.
      Best epoch selected by test mlogloss on held-out game period.

  Policy output
      For every bar in [game_start, game_end] simulate a live portfolio
      and run per-bar inference.  Writes action_id (0-2) + probabilities
      + portfolio snapshot to JSON.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

# ── constants ──────────────────────────────────────────────────────────────────
DATE_COLUMN    = "Date"
N_ACTIONS      = 3
ACTIONS        = {0: "HOLD", 1: "BUY", 2: "SELL"}
COMMISSION     = 0.001       # 0.1% per-side transaction cost
LONG_WINDOW    = 60          # long context window (bars)
MAX_BUY_SHARES = 100         # max shares per single BUY order
MAX_LOTS       = 50          # max concurrent open lots
N_MKTFEATS     = 124         # market feature dims (build_bar_features output)
N_PORTFEATS    = 5           # portfolio state feature dims
N_FEATURES     = N_MKTFEATS + N_PORTFEATS   # 129 total model input dims


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
                         label=np.random.randint(0, N_ACTIONS, 64).astype(np.int32))
        xgb.train({"objective": "multi:softprob", "num_class": N_ACTIONS,
                   "tree_method": "hist", "device": "cuda", "max_depth": 2},
                  dm, num_boost_round=2, verbose_eval=False)
        return "cuda"
    except Exception:
        return "cpu"


# ── labeling ───────────────────────────────────────────────────────────────────
def label_by_return(ret: float, th_lo: float, th_hi: float = 0.008) -> int:
    """3-class oracle: BUY(1) if ret>th_lo, SELL(2) if ret<-th_lo, else HOLD(0)."""
    if ret > th_lo:    return 1   # BUY
    if ret < -th_lo:   return 2   # SELL
    return 0                       # HOLD


# ── portfolio state features ───────────────────────────────────────────────────
def portfolio_features(cash: float, lots: list, lot_costs: list,
                        price: float, initial_cash: float) -> np.ndarray:
    """5 scale-invariant portfolio state features.

    Dims:
      0 – cash_ratio      : cash / total_equity
      1 – invested_ratio  : market_value / total_equity
      2 – position_size   : total_shares / (MAX_BUY_SHARES * MAX_LOTS)
      3 – equity_growth   : tanh(equity / initial_cash − 1)
      4 – cost_vs_price   : avg_cost / price − 1  (negative = in profit)
    """
    n_shares = float(sum(lots))
    invested = n_shares * price
    total    = cash + invested
    if lots:
        avg_cost = float(sum(q * c for q, c in zip(lots, lot_costs)) / n_shares)
    else:
        avg_cost = float(price)
    return np.array([
        cash     / max(total, 1e-9),
        invested / max(total, 1e-9),
        n_shares / max(float(MAX_BUY_SHARES * MAX_LOTS), 1.0),
        float(np.tanh(total / max(initial_cash, 1e-9) - 1.0)),
        avg_cost / max(float(price), 1e-9) - 1.0,
    ], dtype=np.float32)


def neutral_portfolio_features() -> np.ndarray:
    """All-cash, no-position portfolio state: [1, 0, 0, 0, 0]."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ── Phase 1: random exploration ────────────────────────────────────────────────
def _build_valid_bars(dates, values, close_idx, high_idx, low_idx,
                      train_start, train_end, window):
    """Return list of (bar_idx, features[121-dim], real_price, momentum_ret).
    Requires at least LONG_WINDOW bars of history before the first valid bar.
    """
    LOOKBACK  = min(5, window - 1)
    min_start = max(window, LONG_WINDOW) - 1
    result    = []
    for i in range(min_start, len(dates)):
        d = dates[i].date()
        if d < train_start or d >= train_end:
            continue
        raw14 = values[i - window + 1 : i + 1]
        raw60 = values[i - LONG_WINDOW + 1 : i + 1]
        feat  = build_bar_features(raw14, raw60, dates[i],
                                   close_idx, high_idx, low_idx)
        px     = float(values[i, close_idx])
        c_now  = float(values[i, close_idx])
        c_prev = float(values[i - LOOKBACK, close_idx])
        ret    = (c_now - c_prev) / max(abs(c_prev), 1e-9)
        result.append((i, feat, px, ret))
    return result


def _run_single_exploration(valid_bars, start_idx, episode_len, start_cash, rng,
                            max_lots=MAX_LOTS, th_lo=0.003, th_hi=0.008):
    """One random-action episode with portfolio tracking.
    records: list of (feat_129, oracle_class).
    Oracle label via label_by_return — NOT the action taken.
    Includes 0.1% per-side transaction commission.
    """
    cash      = float(start_cash)
    lots      = []        # share quantities per lot
    lot_costs = []        # purchase prices (parallel to lots)
    records   = []
    end_idx   = min(start_idx + episode_len, len(valid_bars))

    for j in range(start_idx, end_idx):
        _, market_feat, px, ret = valid_bars[j]
        port_feat = portfolio_features(cash, lots, lot_costs, px, start_cash)
        full_feat = np.concatenate([market_feat, port_feat]).astype(np.float32)

        action = int(rng.integers(0, N_ACTIONS))
        qty    = int(rng.integers(1, MAX_BUY_SHARES + 1))   # random quantity

        if action == 1:  # BUY
            cost = px * qty * (1.0 + COMMISSION)
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
                lot_costs.append(float(px))
            else:
                action = 0
        elif action == 2:  # SELL
            total_shares = sum(lots)
            if total_shares > 0:
                shares_to_sell = min(qty, total_shares)
                while shares_to_sell > 0 and lots:
                    if lots[0] <= shares_to_sell:
                        shares_to_sell -= lots[0]
                        cash += lots.pop(0) * px * (1.0 - COMMISSION)
                        lot_costs.pop(0)
                    else:
                        lots[0] -= shares_to_sell
                        cash += shares_to_sell * px * (1.0 - COMMISSION)
                        shares_to_sell = 0
            else:
                action = 0

        records.append((full_feat, label_by_return(ret, th_lo, th_hi)))  # oracle class

    final_px  = valid_bars[end_idx - 1][2] if end_idx > start_idx else 0.0
    equity    = cash + sum(q * final_px for q in lots)
    ep_return = (equity - start_cash) / max(start_cash, 1e-9)
    return records, ep_return


def run_random_exploration(dates, values, close_idx, high_idx, low_idx,
                           train_start, train_end, window,
                           n_episodes, episode_len, capital_pool, rng,
                           th_lo=0.003, th_hi=0.008):
    """Run n_episodes random episodes.  Returns (X, y_oracle_cls, weights) or None."""
    valid = _build_valid_bars(dates, values, close_idx, high_idx, low_idx,
                              train_start, train_end, window)
    if not valid:
        return None, None, None

    max_start = max(1, len(valid) - episode_len)
    all_x, all_y, all_w = [], [], []
    for _ in range(n_episodes):
        start_idx  = int(rng.integers(0, max_start))
        start_cash = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
        records, ep_ret = _run_single_exploration(
            valid, start_idx, episode_len, start_cash, rng, th_lo=th_lo, th_hi=th_hi)
        w = float(np.clip(1.0 + 5.0 * ep_ret, 0.1, 6.0))
        for feat, lbl in records:
            all_x.append(feat)
            all_y.append(lbl)    # int oracle class (0-5)
            all_w.append(w)

    if not all_x:
        return None, None, None

    return (np.vstack(all_x).astype(np.float32),
            np.array(all_y, dtype=np.int32),
            np.array(all_w, dtype=np.float32))


# ── feature normalization & engineering ──────────────────────────────────────
def window_to_feats(window_data: np.ndarray) -> np.ndarray:
    """Per-window z-score normalization — scale and level invariant."""
    mean = window_data.mean(axis=0, keepdims=True)
    std  = window_data.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    return ((window_data - mean) / std).astype(np.float32).reshape(-1)


def compute_rsi(close: np.ndarray) -> float:
    """Simple RSI, returned as scalar in [-1, 1] (center-normalized)."""
    if len(close) < 2:
        return 0.0
    diffs  = np.diff(close.astype(np.float64))
    gains  = diffs.clip(min=0.0).mean()
    losses = (-diffs).clip(min=0.0).mean()
    if losses < 1e-9:
        return 1.0
    rs  = gains / losses
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return float((rsi - 50.0) / 50.0)   # [-1, 1]


def compute_atr_ratio(ohlcv: np.ndarray, h_idx: int, l_idx: int, c_idx: int) -> float:
    """ATR(n) / close — scale-invariant volatility measure."""
    trs = []
    for k in range(1, len(ohlcv)):
        h  = float(ohlcv[k, h_idx])
        l  = float(ohlcv[k, l_idx])
        pc = float(ohlcv[k - 1, c_idx])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if not trs:
        return 0.0
    return float(np.mean(trs) / max(abs(float(ohlcv[-1, c_idx])), 1e-9))


def time_features(d) -> np.ndarray:
    """Cyclical time encoding: [sin_dow, cos_dow, sin_doy, cos_doy, year_norm]."""
    if hasattr(d, "date"):
        d = d.date()
    dow = d.weekday()                          # 0=Mon … 4=Fri
    doy = d.timetuple().tm_yday                # 1 … 366
    return np.array([
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
        float(d.year - 2000) / 30.0,           # ~0–1 over 2000-2030
    ], dtype=np.float32)


def build_bar_features(window14: np.ndarray, window60: np.ndarray,
                        date, close_idx: int, high_idx: int, low_idx: int) -> np.ndarray:
    """Full feature vector for one bar (121 dims).

    Dims:
      98 – per-window z-score of 14-bar OHLCV block
       1 – RSI(14) in [-1, 1]
       1 – ATR(14) / close  (scale-invariant volatility)
      13 – log daily returns within 14-bar window
       1 – 60-bar momentum   (close[last]/close[first] − 1)
       1 – 60-bar return volatility
       1 – 60-bar range position  (close in [min60, max60])
       5 – time [sin_dow, cos_dow, sin_doy, cos_doy, year_norm]
       3 – MA3/close−1, MA5/close−1, MA14/close−1  (scale-invariant)
     ═══
     124 total
    """
    # 1. Per-window z-score of raw 14-bar block (98 dims)
    feat14 = window_to_feats(window14)

    # 2. RSI and ATR from 14-bar window
    close14 = window14[:, close_idx].astype(np.float64)
    rsi     = compute_rsi(close14)
    atr_r   = compute_atr_ratio(window14, high_idx, low_idx, close_idx)

    # 3. Log daily returns inside 14-bar window (13 values)
    log_rets14 = np.diff(np.log(np.maximum(close14, 1e-9))).astype(np.float32)

    # 4. 60-bar macro summary
    close60    = window60[:, close_idx].astype(np.float64)
    mom60      = float((close60[-1] - close60[0]) / max(abs(close60[0]), 1e-9))
    log_rets60 = np.diff(np.log(np.maximum(close60, 1e-9)))
    vol60      = float(np.std(log_rets60)) if len(log_rets60) > 1 else 0.0
    mn60, mx60 = float(close60.min()), float(close60.max())
    rng60      = float((close60[-1] - mn60) / max(mx60 - mn60, 1e-9))

    # 5. Cyclical time encoding (5 dims)
    t_feats = time_features(date)

    # 6. MA3, MA5, MA14 relative to current close (3 dims, scale-invariant)
    cur_close = float(close14[-1])
    ma3  = float(close14[-3:].mean())  / max(abs(cur_close), 1e-9) - 1.0
    ma5  = float(close14[-5:].mean())  / max(abs(cur_close), 1e-9) - 1.0
    ma14 = float(close14.mean())       / max(abs(cur_close), 1e-9) - 1.0

    return np.concatenate([
        feat14,
        [rsi, atr_r],
        log_rets14,
        [mom60, vol60, rng60],
        t_feats,
        [ma3, ma5, ma14],
    ]).astype(np.float32)



def build_supervised_dataset(dates, values, close_idx, high_idx, low_idx,
                              train_start, train_end, window, horizon,
                              th_lo, th_hi, label_mode="momentum"):
    """Build supervised training samples with INT class labels (0-5).

    label_mode='momentum' : 5-day trailing momentum return → class via label_by_return
    label_mode='forward'  : horizon-day forward return → class via label_by_return
    Returns (X float32, y int32, idx int32)
    """
    n        = len(dates)
    LOOKBACK = min(5, window - 1)
    min_start = max(window, LONG_WINDOW) - 1
    xs, ys, idx_arr = [], [], []

    for i in range(min_start, n):
        d0 = dates[i].date()
        if d0 < train_start or d0 >= train_end:
            continue
        if label_mode == "forward":
            if i + horizon >= n:
                continue
            d1 = dates[i + horizon].date()
            if d1 >= train_end:
                continue
        raw14 = values[i - window + 1 : i + 1]
        raw60 = values[i - LONG_WINDOW + 1 : i + 1]
        feat  = build_bar_features(raw14, raw60, dates[i],
                                   close_idx, high_idx, low_idx)
        if label_mode == "forward":
            c0  = float(values[i, close_idx])
            c1  = float(values[i + horizon, close_idx])
            ret = (c1 - c0) / max(abs(c0), 1e-9)
        else:
            c_now  = float(values[i, close_idx])
            c_prev = float(values[i - LOOKBACK, close_idx])
            ret    = (c_now - c_prev) / max(abs(c_prev), 1e-9)
        xs.append(feat)
        ys.append(label_by_return(ret, th_lo, th_hi))   # int class label 0-5
        idx_arr.append(i)

    if not xs:
        raise RuntimeError("No supervised samples built.")
    return (np.vstack(xs).astype(np.float32),
            np.array(ys, dtype=np.int32),
            np.array(idx_arr, dtype=np.int32))


# ── portfolio backtest ─────────────────────────────────────────────────────────
def portfolio_backtest(model, x_market_va, row_indices, values, close_idx,
                       start_cash, max_lots=MAX_LOTS, max_buy_shares=MAX_BUY_SHARES):
    """Sequential backtest with live portfolio features and dynamic quantities.

    model        : trained XGBoost model (expects N_FEATURES=126-dim input)
    x_market_va  : (N, N_MKTFEATS=124) pre-computed market features
    row_indices  : corresponding bar indices into values array

    For speed, we batch-predict using neutral portfolio features then update
    sequentially — a close approximation that avoids per-bar DMatrix creation.
    """
    cash      = float(start_cash)
    lots      = []
    lot_costs = []

    # Batch predict: append neutral portfolio once, predict all rows at once.
    nport      = np.tile(neutral_portfolio_features(), (len(x_market_va), 1))
    x_full_all = np.hstack([x_market_va, nport]).astype(np.float32)
    all_probs  = model.predict(xgb.DMatrix(x_full_all)).reshape(-1, N_ACTIONS)

    for step, (idx, probs) in enumerate(zip(row_indices, all_probs)):
        px        = float(values[idx, close_idx])
        action    = int(np.argmax(probs))

        if action == 1:  # BUY
            qty  = max(1, round(float(probs[1]) * max_buy_shares))
            cost = px * qty * (1.0 + COMMISSION)
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
                lot_costs.append(float(px))
        elif action == 2:  # SELL
            total_shares = sum(lots)
            if total_shares > 0:
                shares_to_sell = max(1, round(float(probs[2]) * total_shares))
                while shares_to_sell > 0 and lots:
                    if lots[0] <= shares_to_sell:
                        shares_to_sell -= lots[0]
                        cash += lots.pop(0) * px * (1.0 - COMMISSION)
                        lot_costs.pop(0)
                    else:
                        lots[0] -= shares_to_sell
                        cash += shares_to_sell * px * (1.0 - COMMISSION)
                        shares_to_sell = 0

    final_px = float(values[row_indices[-1], close_idx])
    equity   = cash + sum(q * final_px for q in lots)
    return {
        "final_equity": float(equity),
        "pnl":          float(equity - start_cash),
        "return_pct":   float((equity - start_cash) / max(start_cash, 1e-9) * 100.0),
        "open_shares":  int(sum(lots)),
    }


# ── model training ─────────────────────────────────────────────────────────────
def time_split(x, y, idx, w, valid_ratio=0.2):
    n = x.shape[0]
    split = max(1, min(n - 1, int(n * (1.0 - valid_ratio))))
    def _s(arr):
        return arr[:split], arr[split:]
    return (*_s(x), *_s(y), *_s(idx), *_s(w))


def train_candidate(x_tr, y_tr, w_tr, x_va, y_va, params, num_boost_round):
    """Train one XGB classifier; returns (model, pred_cls_va, val_mlogloss)."""
    dtr   = xgb.DMatrix(x_tr, label=y_tr, weight=w_tr)
    dva   = xgb.DMatrix(x_va, label=y_va)
    model = xgb.train(
        params, dtr,
        num_boost_round=num_boost_round,
        evals=[(dtr, "train"), (dva, "valid")],
        verbose_eval=False,
    )
    probs        = model.predict(dva).reshape(-1, N_ACTIONS)
    pred_cls     = np.argmax(probs, axis=1).astype(np.int32)
    n            = len(y_va)
    val_mlogloss = float(-np.mean(np.log(probs[np.arange(n), y_va.astype(int)] + 1e-9)))
    return model, pred_cls, val_mlogloss


def optimize_model(x_sup_tr, y_sup_tr, w_sup_tr,
                   x_sup_va, y_sup_va, idx_sup_va,
                   x_exp_tr, y_exp_tr, w_exp_tr,
                   values, close_idx, device, num_boost_round,
                   capital_pool, eval_episodes, rng,
                   th_lo=0.003, th_hi=0.008, loss="mae"):
    # Class-frequency weighting for supervised anchor
    sup_cls  = y_sup_tr.astype(np.int32)
    cls_cnts = np.bincount(sup_cls, minlength=N_ACTIONS).astype(np.float32)
    cls_w    = len(sup_cls) / (float(N_ACTIONS) * np.maximum(cls_cnts, 1.0))
    sup_w_bal = cls_w[sup_cls]

    # Add neutral portfolio features to supervised data (121 → 126 dims)
    nport_tr      = np.tile(neutral_portfolio_features(), (len(x_sup_tr), 1))
    nport_va      = np.tile(neutral_portfolio_features(), (len(x_sup_va), 1))
    x_sup_tr_full = np.hstack([x_sup_tr, nport_tr])
    x_sup_va_full = np.hstack([x_sup_va, nport_va])

    # Combine exploration (126-dim) + class-balanced supervised (126-dim)
    if x_exp_tr is not None and len(x_exp_tr) > 0:
        x_tr = np.vstack([x_exp_tr, x_sup_tr_full])
        y_tr = np.concatenate([y_exp_tr, y_sup_tr])
        w_tr = np.concatenate([w_exp_tr, sup_w_bal])
    else:
        x_tr, y_tr, w_tr = x_sup_tr_full, y_sup_tr, sup_w_bal

    x_va, y_va = x_sup_va_full, y_sup_va
    idx_va     = idx_sup_va

    objective   = "multi:softprob"
    eval_metric = "mlogloss"

    candidates = [
        {"eta": 0.03, "max_depth": 8,  "subsample": 0.9,  "colsample_bytree": 0.9,  "min_child_weight": 2},
        {"eta": 0.05, "max_depth": 10, "subsample": 0.9,  "colsample_bytree": 0.9,  "min_child_weight": 3},
        {"eta": 0.07, "max_depth": 12, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4},
    ]

    best       = None
    all_scores = []
    for i, g in enumerate(candidates, 1):
        params = {
            "objective":   objective,
            "eval_metric": eval_metric,
            "num_class":   N_ACTIONS,
            "tree_method": "hist",
            "device":      device,
            "seed":        2026 + i,
            **g,
        }
        model, pred_cls, val_mlogloss = train_candidate(
            x_tr, y_tr, w_tr, x_va, y_va, params, num_boost_round)

        ep_returns = []
        for _ in range(eval_episodes):
            sc   = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
            perf = portfolio_backtest(model, x_sup_va, idx_va, values, close_idx,
                                      start_cash=sc)
            ep_returns.append(perf["return_pct"])

        row = {
            "candidate": i, "params": g,
            "val_mlogloss": float(val_mlogloss),
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
                                 start_cash, use_greedy, current_model, rng,
                                 th_lo=0.003, th_hi=0.008, max_lots=MAX_LOTS,
                                 max_buy_shares=MAX_BUY_SHARES):
    """Run one RL episode with live portfolio features.

    use_greedy    : if True, use current_model for per-bar inference
    current_model : trained XGBoost model (expects N_FEATURES=126-dim input)
    Returns (records, ep_return, ep_sharpe)
    records: list of (feat_126, oracle_class)
    """
    end_idx       = min(start_idx + episode_len, len(valid_bars))
    cash          = float(start_cash)
    lots          = []
    lot_costs     = []
    records       = []
    equity_series = [float(start_cash)]

    for j in range(start_idx, end_idx):
        _, market_feat, px, ret = valid_bars[j]
        port_feat = portfolio_features(cash, lots, lot_costs, px, start_cash)
        full_feat = np.concatenate([market_feat, port_feat]).astype(np.float32)

        if use_greedy and current_model is not None:
            probs  = current_model.predict(
                xgb.DMatrix(full_feat.reshape(1, -1))
            ).reshape(N_ACTIONS)
            action = int(np.argmax(probs))
            p_act  = float(probs[action])
        else:
            probs  = None
            action = int(rng.integers(0, N_ACTIONS))
            p_act  = 0.5   # neutral confidence for random action

        qty = max(1, round(p_act * max_buy_shares))  # confidence-scaled quantity

        if action == 1:  # BUY
            cost = px * qty * (1.0 + COMMISSION)
            if cost <= cash and len(lots) < max_lots:
                cash -= cost
                lots.append(qty)
                lot_costs.append(float(px))
            else:
                action = 0
        elif action == 2:  # SELL
            total_shares = sum(lots)
            if total_shares > 0:
                shares_to_sell = max(1, round(p_act * total_shares))
                while shares_to_sell > 0 and lots:
                    if lots[0] <= shares_to_sell:
                        shares_to_sell -= lots[0]
                        cash += lots.pop(0) * px * (1.0 - COMMISSION)
                        lot_costs.pop(0)
                    else:
                        lots[0] -= shares_to_sell
                        cash += shares_to_sell * px * (1.0 - COMMISSION)
                        shares_to_sell = 0
            else:
                action = 0

        records.append((full_feat, label_by_return(ret, th_lo, th_hi)))  # oracle class
        equity_series.append(cash + sum(q * px for q in lots))

    final_px  = valid_bars[end_idx - 1][2] if end_idx > start_idx else 0.0
    equity    = cash + sum(q * final_px for q in lots)
    ep_return = (equity - float(start_cash)) / max(float(start_cash), 1e-9)

    eq_arr    = np.array(equity_series, dtype=np.float64)
    step_rets = np.diff(np.log(np.maximum(eq_arr, 1e-9)))
    ep_sharpe = (float(step_rets.mean()) / (float(step_rets.std()) + 1e-9)
                 * np.sqrt(252)) if len(step_rets) > 1 else 0.0

    return records, ep_return, ep_sharpe


def run_rl_epoch(valid_bars, current_model, epsilon,
                 n_episodes, episode_len, capital_pool, rng,
                 th_lo=0.003, th_hi=0.008):
    """One RL epoch: each episode is 100% random OR 100% greedy (portfolio-aware).
    epsilon = fraction of episodes that are pure random exploration.
    """
    n_bars    = len(valid_bars)
    max_start = max(1, n_bars - episode_len)
    all_x, all_y, all_w, returns = [], [], [], []
    n_random  = int(round(epsilon * n_episodes))

    for ep_i in range(n_episodes):
        use_greedy = (ep_i >= n_random) and (current_model is not None)
        start_idx  = int(rng.integers(0, max_start))
        start_cash = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
        records, ep_ret, ep_sharpe = _run_policy_episode_segment(
            valid_bars, start_idx, episode_len, start_cash,
            use_greedy, current_model, rng, th_lo=th_lo, th_hi=th_hi,
        )
        w = float(np.clip(0.5 + 1.5 * ep_sharpe + 3.0 * ep_ret, 0.1, 6.0))
        for feat, lbl in records:
            all_x.append(feat)
            all_y.append(lbl)
            all_w.append(w)
        returns.append(ep_ret)

    return all_x, all_y, all_w, returns


def run_rl_training(valid_bars,
                    x_sup_tr, y_sup_tr,
                    x_sup_va, y_sup_va, idx_sup_va,
                    x_test, y_test,
                    values, close_idx, device, num_boost_round,
                    capital_pool, eval_episodes, rng,
                    best_params, n_epochs, n_episodes_per_epoch,
                    episode_len, eps_start, eps_end,
                    th_lo=0.003, th_hi=0.008, loss="mae",
                    max_ep_ratio=5):
    """Iterative policy improvement (3-class multi:softprob).

    x_sup_tr/x_sup_va/x_test: 121-dim market features.
    Neutral portfolio features are appended here before training.
    Episode data (126-dim) is generated with live portfolio features.
    Best epoch selected by test mlogloss on held-out game period.
    """
    best_model         = None
    best_test_mlogloss = np.inf
    best_val_ret       = 0.0
    best_epoch         = 0
    max_ep_samples     = len(y_sup_tr) * max_ep_ratio

    objective   = "multi:softprob"
    eval_metric = "mlogloss"

    # Append neutral portfolio features to supervised/test data (121 → 126 dims)
    nport_tr   = np.tile(neutral_portfolio_features(), (len(x_sup_tr), 1))
    nport_va   = np.tile(neutral_portfolio_features(), (len(x_sup_va), 1))
    nport_te   = np.tile(neutral_portfolio_features(), (len(x_test),   1))
    x_sup_tr_f = np.hstack([x_sup_tr, nport_tr])
    x_sup_va_f = np.hstack([x_sup_va, nport_va])
    x_test_f   = np.hstack([x_test,   nport_te])

    # Class-frequency weights for supervised anchor (inverse freq per bucket)
    sup_cls  = y_sup_tr.astype(np.int32)
    cls_cnts = np.bincount(sup_cls, minlength=N_ACTIONS).astype(np.float32)
    cls_w    = len(sup_cls) / (float(N_ACTIONS) * np.maximum(cls_cnts, 1.0))
    sup_w    = cls_w[sup_cls]

    base_params = {
        "objective":   objective,
        "eval_metric": eval_metric,
        "num_class":   N_ACTIONS,
        "tree_method": "hist",
        "device":      device,
        **best_params,
    }

    dtest = xgb.DMatrix(x_test_f, label=y_test)

    for epoch in range(n_epochs):
        if epoch == 0:
            epsilon = 1.0
        elif n_epochs > 2:
            t       = (epoch - 1) / max(n_epochs - 2, 1)
            epsilon = eps_start - (eps_start - eps_end) * t
        else:
            epsilon = eps_end

        print(f"\nEpoch {epoch + 1}/{n_epochs}  epsilon={epsilon:.3f}", flush=True)

        ep_x, ep_y, ep_w, returns = run_rl_epoch(
            valid_bars, best_model, epsilon,
            n_episodes_per_epoch, episode_len, capital_pool, rng,
            th_lo=th_lo, th_hi=th_hi,
        )

        # Cap exploration samples so supervised signal isn't drowned out
        ep_x_arr = np.vstack(ep_x).astype(np.float32)   # already 126-dim
        ep_y_arr = np.array(ep_y, dtype=np.int32)
        ep_w_arr = np.array(ep_w, dtype=np.float32)
        if len(ep_y_arr) > max_ep_samples:
            sel      = rng.choice(len(ep_y_arr), size=max_ep_samples, replace=False)
            ep_x_arr = ep_x_arr[sel]
            ep_y_arr = ep_y_arr[sel]
            ep_w_arr = ep_w_arr[sel]

        n_rand_ep = int(round(epsilon * n_episodes_per_epoch))
        n_xgb_ep  = n_episodes_per_epoch - n_rand_ep
        print(f"  random_eps={n_rand_ep}  xgb_eps={n_xgb_ep}"
              f"  avg_ep_return={np.mean(returns)*100:.2f}%"
              f"  ep_samples_used={len(ep_y_arr)}", flush=True)

        # Combine: supervised (neutral portfolio, 126-dim) + episodes (live portfolio, 126-dim)
        x_tr = np.vstack([x_sup_tr_f, ep_x_arr])
        y_tr = np.concatenate([y_sup_tr, ep_y_arr])
        w_tr = np.concatenate([sup_w, ep_w_arr])

        params = {**base_params, "seed": 2026 + epoch}
        dtrain = xgb.DMatrix(x_tr, label=y_tr, weight=w_tr)
        dva    = xgb.DMatrix(x_sup_va_f, label=y_sup_va)
        model  = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dva, "valid")],
            verbose_eval=False,
        )

        # Validation mlogloss
        probs_va     = model.predict(dva).reshape(-1, N_ACTIONS)
        nva          = len(y_sup_va)
        val_mlogloss = float(-np.mean(np.log(
            probs_va[np.arange(nva), y_sup_va.astype(int)] + 1e-9)))

        # Test mlogloss on game period (primary selection criterion)
        probs_test    = model.predict(dtest).reshape(-1, N_ACTIONS)
        nte           = len(y_test)
        test_mlogloss = float(-np.mean(np.log(
            probs_test[np.arange(nte), y_test.astype(int)] + 1e-9)))

        # Live portfolio backtest on validation bars
        ep_returns = []
        for _ in range(eval_episodes):
            sc   = float(capital_pool[int(rng.integers(0, len(capital_pool)))])
            perf = portfolio_backtest(
                model, x_sup_va, idx_sup_va, values, close_idx, start_cash=sc)
            ep_returns.append(perf["return_pct"])
        avg_ret = float(np.mean(ep_returns))

        marker = ""
        if test_mlogloss < best_test_mlogloss:
            best_test_mlogloss = test_mlogloss
            best_val_ret       = avg_ret
            best_model         = model
            best_epoch         = epoch + 1
            marker             = "  ← best (mlogloss)"
        print(f"  val_mlogloss={val_mlogloss:.5f}  test_mlogloss={test_mlogloss:.5f}  "
              f"val_avg_return={avg_ret:.2f}%  "
              f"[best mlogloss={best_test_mlogloss:.5f} @ epoch {best_epoch}]{marker}",
              flush=True)

    print(f"\nRL training done. Best epoch={best_epoch}  "
          f"test_mlogloss={best_test_mlogloss:.5f}  val_avg_return={best_val_ret:.2f}%")
    return best_model, best_test_mlogloss, best_val_ret, best_epoch


# ── game policy ────────────────────────────────────────────────────────────────
def build_game_policy_xgb(dates, values, model, close_idx, high_idx, low_idx,
                           game_start, game_end, window, th_lo=0.003, th_hi=0.008):
    """XGB inference for each game bar with live portfolio simulation.

    Portfolio starts at game_start with start_cash (500,000 mid-pool).
    Action quantities are dynamically scaled by softmax confidence.
    """
    start_cash = 500_000.0   # representative mid-pool capital
    n          = len(dates)
    rows       = []
    min_start  = max(window, LONG_WINDOW) - 1
    cash       = float(start_cash)
    lots       = []
    lot_costs  = []

    for i in range(min_start, n):
        d = dates[i].date()
        if d < game_start:
            continue
        if d > game_end:
            break
        raw14    = values[i - window + 1 : i + 1]
        raw60    = values[i - LONG_WINDOW + 1 : i + 1]
        mkt_feat = build_bar_features(raw14, raw60, dates[i], close_idx, high_idx, low_idx)
        px       = float(values[i, close_idx])

        port_feat = portfolio_features(cash, lots, lot_costs, px, start_cash)
        full_feat = np.concatenate([mkt_feat, port_feat]).reshape(1, -1).astype(np.float32)
        probs     = model.predict(xgb.DMatrix(full_feat)).reshape(N_ACTIONS)
        cls       = int(np.argmax(probs))
        prob      = {ACTIONS[k]: float(probs[k]) for k in range(N_ACTIONS)}

        # Execute action to update portfolio state for next bar
        if cls == 1:  # BUY
            qty  = max(1, round(float(probs[1]) * MAX_BUY_SHARES))
            cost = px * qty * (1.0 + COMMISSION)
            if cost <= cash and len(lots) < MAX_LOTS:
                cash -= cost
                lots.append(qty)
                lot_costs.append(px)
            else:
                cls = 0
        elif cls == 2:  # SELL
            total_shares = sum(lots)
            if total_shares > 0:
                shares_to_sell = max(1, round(float(probs[2]) * total_shares))
                while shares_to_sell > 0 and lots:
                    if lots[0] <= shares_to_sell:
                        shares_to_sell -= lots[0]
                        cash += lots.pop(0) * px * (1.0 - COMMISSION)
                        lot_costs.pop(0)
                    else:
                        lots[0] -= shares_to_sell
                        cash += shares_to_sell * px * (1.0 - COMMISSION)
                        shares_to_sell = 0
            else:
                cls = 0

        equity = cash + sum(q * px for q in lots)
        rows.append({
            "date":          d.isoformat(),
            "action_id":     cls,
            "action_name":   ACTIONS[cls],
            "signal":        ("BUY" if cls == 1 else "SELL" if cls == 2 else "HOLD"),
            "pred_class":    cls,
            "pred_return":   float(probs[cls]),
            "probabilities": prob,
            "close":         float(px),
            "portfolio": {
                "cash":        round(cash, 2),
                "equity":      round(equity, 2),
                "shares_held": int(sum(lots)),
                "n_lots":      int(len(lots)),
            },
        })
    if not rows:
        raise RuntimeError("No game policy rows in requested range.")
    return rows


def build_game_policy_momentum(dates, values, close_idx, game_start, game_end,
                                window, lookback, th_lo, th_hi):
    """Rule-based momentum game policy (no future leakage)."""
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
        prob   = {ACTIONS[k]: 0.05 for k in range(N_ACTIONS)}
        prob[ACTIONS[cls]] = 0.90
        rows.append({
            "date":          d.isoformat(),
            "action_id":     cls,
            "action_name":   ACTIONS[cls],
            "signal":        ("BUY" if cls == 1 else "SELL" if cls == 2 else "HOLD"),
            "pred_class":    cls,
            "probabilities": prob,
            "close":         float(values[i, close_idx]),
            "momentum_ret":  round(float(ret), 6),
        })
    if not rows:
        raise RuntimeError("No game policy rows in requested range.")
    return rows


def build_game_policy(dates, values, model, close_idx, high_idx, low_idx,
                      game_start, game_end, window,
                      lookback=5, th_lo=0.003, th_hi=0.008,
                      game_policy_mode="momentum"):
    """Dispatch to XGB regression or momentum game policy."""
    if game_policy_mode == "momentum":
        return build_game_policy_momentum(dates, values, close_idx,
                                          game_start, game_end, window,
                                          lookback, th_lo, th_hi)
    return build_game_policy_xgb(dates, values, model, close_idx, high_idx, low_idx,
                                 game_start, game_end, window,
                                 th_lo=th_lo, th_hi=th_hi)


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
    ap.add_argument("--episode-len", type=int, default=512,
                    help="Steps per episode (default 252 ≈ 1 trading year)")
    ap.add_argument("--eps-start", type=float, default=0.8,
                    help="Epsilon at epoch 1 (epoch 0 is always 1.0). Default 0.8")
    ap.add_argument("--eps-end", type=float, default=0.1,
                    help="Epsilon at final epoch. Default 0.1")
    ap.add_argument("--loss", choices=["mae", "mse"], default="mae",
                    help="Regression loss: mae (reg:absoluteerror) or mse (reg:squarederror). Default mae.")
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
    high_idx  = feature_columns.index("High") if "High" in feature_columns else close_idx
    low_idx   = feature_columns.index("Low")  if "Low"  in feature_columns else close_idx

    device = select_device(args.prefer_cuda)
    print(f"Using per-window normalization (scale-invariant features).")

    # ── Phase 1: random exploration ──────────────────────────────────────────
    print(f"Phase 1: random exploration ({args.explore_episodes} episodes) ...")
    x_exp, y_exp, w_exp = run_random_exploration(
        dates, values, close_idx, high_idx, low_idx,
        train_start, train_end,
        args.window,
        n_episodes=args.explore_episodes,
        episode_len=args.episode_len,
        capital_pool=capital_pool,
        rng=rng,
        th_lo=args.th_lo, th_hi=args.th_hi,
    )
    if x_exp is None:
        print("  (No exploration data generated, skipping phase 1)")

    # ── Phase 2: supervised labels ───────────────────────────────────────────
    print("Phase 2: building supervised labels ...")
    x_sup, y_sup, idx_sup = build_supervised_dataset(
        dates, values, close_idx, high_idx, low_idx,
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
            th_lo=args.th_lo, th_hi=args.th_hi, loss=args.loss,
        )
        best_params = best_classic["params"]
        print(f"  Best hyperparams: {best_params}")

        # Step 2: build valid_bars list for RL episode generation
        valid_bars  = _build_valid_bars(
            dates, values, close_idx, high_idx, low_idx, train_start, train_end, args.window)
        print(f"  Valid bars: {len(valid_bars)}  Market dims: {N_MKTFEATS}  Total dims: {N_FEATURES}")

        # Step 3: build test set from game period (used for test MAE per epoch)
        from datetime import timedelta
        print("  Building test set from game period ...")
        x_test, y_test, idx_test = build_supervised_dataset(
            dates, values, close_idx, high_idx, low_idx,
            game_start, game_end + timedelta(days=1),
            args.window, args.horizon, args.th_lo, args.th_hi,
            label_mode=args.label_mode,
        )
        print(f"  Test samples (game period): {len(y_test)}")

        # Step 4: RL loop
        rl_model, rl_test_mlogloss, rl_val_ret, rl_epoch = run_rl_training(
            valid_bars,
            x_sup_tr, y_sup_tr,
            x_sup_va, y_sup_va, idx_sup_va,
            x_test, y_test,
            values, close_idx, device, args.rounds,
            capital_pool, args.eval_episodes, rng,
            best_params,
            n_epochs=args.rl_epochs,
            n_episodes_per_epoch=args.episodes_per_epoch,
            episode_len=args.episode_len,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            th_lo=args.th_lo,
            th_hi=args.th_hi,
            loss=args.loss,
        )

        model_path = out_dir / "xgb_multibuy_model.json"
        rl_model.save_model(str(model_path))
        best_info = {
            "model":          rl_model,
            "params":         best_params,
            "valid_accuracy": None,
            "avg_return_pct": rl_val_ret,
            "min_return_pct": rl_val_ret,
            "max_return_pct": rl_val_ret,
            "best_rl_epoch":  rl_epoch,
            "best_test_mlogloss": rl_test_mlogloss,
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
            th_lo=args.th_lo, th_hi=args.th_hi, loss=args.loss,
        )
        model_path = out_dir / "xgb_multibuy_model.json"
        best_info["model"].save_model(str(model_path))
        print(f"\nBest candidate: #{best_info['candidate']}")

    print(f"  avg_return_pct = {best_info['avg_return_pct']:.4f}%")

    # ── Generate game policy ─────────────────────────────────────────────────
    print(f"\nGenerating game policy (mode={args.game_policy_mode}) ...")
    lookback = min(5, args.window - 1)
    policy_rows = build_game_policy(
        dates, values, best_info["model"], close_idx, high_idx, low_idx,
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
