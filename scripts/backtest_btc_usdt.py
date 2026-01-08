#!/usr/bin/env python3
"""
BTC/crypto microstructure backtest scaffold with parameter sweeps
================================================================

What this does
--------------
1) Ingest sub-second CSV inputs (ticker bid/ask snapshots and/or trade prints).
2) Build a 1-second price series p_t.
3) For each run configuration:
   - Generate decisions u_t (per second) using a selected strategy.
   - Apply execution delay τ: executed action d_t = u_{t-τ}.
   - Evaluate returns over a fixed horizon H (seconds):
         r_t = (p_{t+H} - p_t) / p_t
   - Compute costs:
         C_t(d_t) = - d_t * r_t + κ|d_t| + λ 1[d_t != d_{t-1}]
     (κ, λ in return units; typically bps/10,000).
   - Compute ε-optimal sets:
         G_t(ε) = { d : C_t(d) <= min_{d'} C_t(d') + ε }
     and degeneracy rates.

Outputs
-------
For each run:
- <out_prefix>__<run_id>_per_second.csv
- <out_prefix>__<run_id>_metrics.json

Additionally, for sweep runs, it writes:
- <out_prefix>_sweep_summary.csv   (one row per run; primary artifact for analysis.py)

Notes
-----
- This script separates measurement (backtest) from interpretation (analysis.py).
- Predictor strategies are intentionally simple baselines; replace with your own causal signal model.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities / Parsing
# -----------------------------

def parse_timestamp_series(s: pd.Series, ts_unit: str) -> pd.DatetimeIndex:
    """
    Parse a timestamp column into pandas DatetimeIndex (UTC).

    Parameters
    ----------
    s : pd.Series
        Timestamp values (epoch or ISO strings).
    ts_unit : str
        If epoch numeric, specify 's', 'ms', 'us', or 'ns'. If ISO strings, set to 'auto'.

    Returns
    -------
    pd.DatetimeIndex
    """
    if ts_unit == "auto":
        dt = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=ts_unit, utc=True)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Found {bad} unparsable timestamps. Check --time_col and --ts_unit.")
    return pd.DatetimeIndex(dt)


def bps_to_return_units(bps: float) -> float:
    """Convert basis points to return units (e.g., 3 bps -> 0.0003)."""
    return float(bps) / 10_000.0


def sha1_short(x: str, n: int = 10) -> str:
    """Short stable run id from a config string."""
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:n]


# -----------------------------
# Data Loading and 1-second series construction
# -----------------------------


# -----------------------------
# Core computations
# -----------------------------

def compute_forward_return(prices_1s: pd.Series, horizon_s: int) -> pd.Series:
    """r_t = (p_{t+H} - p_t) / p_t on a 1-second grid."""
    p = prices_1s.astype(float)
    p_fwd = p.shift(-horizon_s)
    return (p_fwd - p) / p


def compute_costs_for_all_actions(r_t: float, kappa: float) -> Dict[int, float]:
    return {
        -1: (+r_t) + kappa,
        0: 0.0,
        +1: (-r_t) + kappa
    }


def strategy_oracle_greedy(r: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    d = np.zeros(len(r), dtype=int)
    prev = 0
    for i, rt in enumerate(r):
        if np.isnan(rt):
            d[i] = 0
            prev = 0
            continue
        costs = {}
        for a in (-1, 0, +1):
            c = (-a * rt) + (kappa * abs(a)) + (lam * (1 if a != prev else 0))
            costs[a] = c
        best_cost = min(costs.values())
        candidates = [a for a, c in costs.items() if c == best_cost]
        if 0 in candidates:
            best = 0
        else:
            best = min(candidates, key=lambda x: (abs(x), x))
        d[i] = best
        prev = best
    return d


def epsilon_optimal_set(costs: Dict[int, float], eps: float) -> Tuple[int, ...]:
    """Return ε-optimal action set as a sorted tuple."""
    m = min(costs.values())
    return tuple(sorted([d for d, c in costs.items() if c <= m + eps]))


# -----------------------------
# Strategies (examples)
# -----------------------------

def strategy_flat(n: int) -> np.ndarray:
    """Always flat."""
    return np.zeros(n, dtype=np.int8)


def strategy_momentum_threshold(prices_1s: pd.Series, lookback_s: int, threshold: float, tau_s: int = 1) -> np.ndarray:
    """
    Causal momentum: compare current price to price lookback seconds ago.
    threshold in return units.
    """
    p = prices_1s.values.astype(float)
    p_delayed = np.roll(p, tau_s)
    p_delayed[:tau_s] = np.nan
    p_lb = np.roll(p_delayed, lookback_s)
    p_lb[:lookback_s + tau_s] = np.nan
    ret = (p_delayed - p_lb) / p_lb
    u = np.zeros(len(p), dtype=np.int8)
    u[ret > threshold] = 1
    u[ret < -threshold] = -1
    u[np.isnan(ret)] = 0
    return u


def strategy_stable_window_lcb(
    prices_1s: pd.Series,
    horizon_s: int,
    window_s: int,
    beta: float,
    kappa: float,
    tau_s: int = 1,
) -> np.ndarray:
    """
    Windowed stable policy using LCB (Lower Confidence Bound) on forward returns (causal).
    We compute forward returns r_t from prices, then use r_{<=t-1} to decide at time t.
    """
    p = prices_1s.astype(float)
    r = compute_forward_return(p, horizon_s=horizon_s)

    n = len(p)
    u = np.zeros(n, dtype=np.int8)
    # Shift returns by horizon to enforce causality
    r_past = r.shift(horizon_s + 1)
    minp = max(10, window_s // 3) # minimum number of valid observations required to compute the rolling statistics

    roll_mean = r_past.rolling(window=window_s, min_periods=minp).mean()
    roll_std = r_past.rolling(window=window_s, min_periods=minp).std(ddof=1)
    roll_n = r_past.rolling(window=window_s, min_periods=minp).count()

    for start in range(0, n, window_s):
        end = min(start + window_s, n)
        mu = roll_mean.iloc[start]
        sd = roll_std.iloc[start]
        cnt = roll_n.iloc[start]

        if np.isnan(mu) or np.isnan(sd) or cnt <= 1:
            act = 0
        else:
            lcb = mu - beta * sd / np.sqrt(cnt)
            if lcb > kappa:
                act = 1
            elif lcb < -kappa:
                act = -1
            else:
                act = 0

        u[start:end] = act

    return u


# -----------------------------
# Backtest runner
# -----------------------------

@dataclass
class BacktestConfig:
    tau_s: int
    horizon_s: int
    kappa: float
    lam: float
    eps: float
    strategy: str
    strategy_params: Dict[str, Any]


def apply_delay(u: np.ndarray, tau_s: int) -> np.ndarray:
    """Executed action d_t = u_{t-τ}. First τ steps are set to flat (0)."""
    if tau_s <= 0:
        return u.copy()
    d = np.zeros_like(u)
    d[tau_s:] = u[:-tau_s]
    d[:tau_s] = 0
    return d


def compute_cost_series(d: np.ndarray, r: np.ndarray, kappa: float, lam: float) -> np.ndarray:
    """C_t(d_t) = -d_t*r_t + kappa|d_t| + lam*1[d_t != d_{t-1}]"""
    n = len(d)
    cost = np.zeros(n, dtype=float)
    for t in range(n):
        switch = 1.0 if (t > 0 and d[t] != d[t - 1]) else 0.0
        cost[t] = -d[t] * r[t] + kappa * abs(int(d[t])) + lam * switch
    return cost


def generate_decisions(prices_1s: pd.DataFrame, cfg: BacktestConfig) -> np.ndarray:
    n = len(prices_1s)
    p = prices_1s["p"].astype(float)

    if cfg.strategy == "flat":
        return strategy_flat(n)

    if cfg.strategy == "mom":
        lookback_s = int(cfg.strategy_params.get("lookback_s", 30))
        threshold = float(cfg.strategy_params.get("threshold", bps_to_return_units(2.0)))
        return strategy_momentum_threshold(p, lookback_s=lookback_s, threshold=threshold, tau_s=cfg.tau_s)

    if cfg.strategy == "oracle":
        r = compute_forward_return(p, horizon_s=cfg.horizon_s).to_numpy()
        return strategy_oracle_greedy(r, kappa=cfg.kappa, lam=cfg.lam)

    if cfg.strategy == "stable_lcb":
        window_s = int(cfg.strategy_params.get("window_s", 30))
        beta = float(cfg.strategy_params.get("beta", 2.0))
        return strategy_stable_window_lcb(
            p,
            horizon_s=cfg.horizon_s,
            window_s=window_s,
            beta=beta,
            kappa=cfg.kappa,
            tau_s=cfg.tau_s,
        )

    raise ValueError(f"Unknown strategy: {cfg.strategy}")


def run_backtest(prices_1s: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run one configuration and return per-second dataframe and summary metrics."""
    if "p" not in prices_1s.columns:
        raise ValueError("prices_1s must include column 'p'.")

    p = prices_1s["p"].astype(float)
    r = compute_forward_return(p, horizon_s=cfg.horizon_s)

    u = generate_decisions(prices_1s, cfg)

    df = prices_1s.copy()
    df["r"] = r
    df["u"] = u

    if cfg.strategy == "oracle":
        d = u.copy()
    else:
        d = apply_delay(u, cfg.tau_s)
    df["d"] = d

    # Keep only rows with defined forward returns (need p_{t+H})
    df = df.loc[df["r"].notna()].copy()

    rv = df["r"].values.astype(float)
    dv = df["d"].values.astype(np.int8)

    cost = compute_cost_series(dv, rv, kappa=cfg.kappa, lam=cfg.lam)
    df["cost"] = cost
    df["pnl"] = -df["cost"]

    # Degeneracy metrics
    gsets: List[str] = []
    gsize = np.zeros(len(df), dtype=int)
    flat_in = np.zeros(len(df), dtype=int)

    for i, rt in enumerate(rv):
        costs_all = compute_costs_for_all_actions(rt, kappa=cfg.kappa)
        g = epsilon_optimal_set(costs_all, eps=cfg.eps)
        gsets.append(str(g))
        gsize[i] = len(g)
        flat_in[i] = 1 if 0 in g else 0

    df["G_eps"] = gsets
    df["G_size"] = gsize
    df["flat_in_G"] = flat_in
    df["degenerate_ge2"] = (df["G_size"] >= 2).astype(int)

    turnover = int(np.sum(dv[1:] != dv[:-1])) if len(dv) > 1 else 0

    # A simple drawdown metric on cumulative pnl
    c = df["pnl"].cumsum()
    dd = (c - c.cummax()).min() if len(c) else 0.0

    metrics = {
        "run_id": None,  # set by caller
        "T": int(len(df)),
        "cum_pnl": float(df["pnl"].sum()),
        "mean_pnl_per_step": float(df["pnl"].mean()) if len(df) else float("nan"),
        "std_pnl_per_step": float(df["pnl"].std(ddof=1)) if len(df) > 1 else float("nan"),
        "turnover": turnover,
        "turnover_per_step": float(turnover / max(1, len(df))),
        "max_drawdown": float(dd),
        "p_flat_in_G": float(df["flat_in_G"].mean()) if len(df) else float("nan"),
        "p_degenerate_ge2": float(df["degenerate_ge2"].mean()) if len(df) else float("nan"),
        "tau_s": int(cfg.tau_s),
        "horizon_s": int(cfg.horizon_s),
        "kappa": float(cfg.kappa),
        "lam": float(cfg.lam),
        "eps": float(cfg.eps),
        "strategy": cfg.strategy,
        "strategy_params": json.dumps(cfg.strategy_params, sort_keys=True),
    }
    return df, metrics


def plot_decisions_vs_price(df: pd.DataFrame, out_path: str, title: str = "Strategy Decisions vs Price") -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax1.plot(df.index, df["p"], label="price", alpha=0.8)
    ax1.set_ylabel("Price")
    ax1.legend()

    ax2.plot(df.index, df["d"], label="decision", drawstyle="steps-post")
    ax2.set_ylabel("Decision (-1, 0, +1)")
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()

    cum_pnl = df["pnl"].cumsum()
    ax3.plot(df.index, cum_pnl, label="cumulative PnL", color="green")
    ax3.set_ylabel("Cumulative PnL")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.legend()

    plt.xlabel("Time")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    #plt.show()


# -----------------------------
# Sweeps and CLI
# -----------------------------

def parse_list_int(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: Optional[str]) -> Optional[List[float]]:
    if s is None or s.strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1-second delayed-horizon backtest with parameter sweeps.")
    p.add_argument("--trades_csv", type=str, default=None, help="Path to trades CSV (sub-second).")
    p.add_argument("--ticker_csv", type=str, default=None, help="Path to ticker/quotes CSV (sub-second).")

    p.add_argument("--time_col", type=str, default='timestamp', help="Timestamp column name in CSV(s).")
    p.add_argument("--ts_unit", type=str, default="ms", choices=["s", "ms", "us", "ns", "auto"],
                   help="Timestamp unit for epoch timestamps, or 'auto' for ISO parsing.")

    # Trades schema
    p.add_argument("--price_col", type=str, default=None, help="Trade price column name (required if --trades_csv).")

    # Ticker schema
    p.add_argument("--bid_col", type=str, default=None, help="Best bid column name (required if --ticker_csv).")
    p.add_argument("--ask_col", type=str, default=None, help="Best ask column name (required if --ticker_csv).")

    # Core backtest parameters
    p.add_argument("--horizon", type=int, default=600, help="Evaluation horizon H in seconds.")
    # default kappa is 10 bps, which is 0.1% buy or sell fee
    p.add_argument("--kappa_bps", type=float, default=10.0, help="Execution cost κ in basis points.")
    p.add_argument("--lam_bps", type=float, default=0.0, help="Switching penalty λ in basis points.")
    p.add_argument("--epsilon_mult", type=float, default=1.0, help="ε = epsilon_mult * κ.")

    # Strategies (single run defaults)
    p.add_argument("--strategy", type=str, default="stable_lcb",
                   choices=["flat", "mom", "stable_lcb", "oracle"],
                   help="Strategy for a single run (no sweep).")
    p.add_argument("--mom_lookback", type=int, default=30, help="Momentum lookback in seconds (mom strategy).")
    p.add_argument("--mom_threshold_bps", type=float, default=2.0, help="Momentum threshold in bps (mom strategy).")
    p.add_argument("--window", type=int, default=150, help="Window length W in seconds (stable_lcb strategy).")
    p.add_argument("--beta", type=float, default=2.0, help="LCB confidence parameter beta (stable_lcb strategy).")
    p.add_argument("--tau", type=int, default=1, help="Execution delay τ in seconds (single run).")

    # Sweep controls (comma-separated lists). If any sweep_* is provided, a sweep run is performed.
    p.add_argument("--sweep_tau", type=str, default="0,1,2,5", help="Comma-separated τ values, e.g. '0,1,2,5'.")
    p.add_argument("--sweep_kappa_bps", type=str, default='2,10,15', help="Comma-separated κ (bps), e.g. '1,2,3,5,10,15'.")
    p.add_argument("--sweep_horizon", type=str, default='120,300,600,3600', help="Comma-separated horizon values in seconds. Window W = horizon/4. e.g. '120,300,600,3600'.")
    p.add_argument("--sweep_epsilon_mult", type=str, default=None, help="Comma-separated ε/κ multipliers, e.g. '1,2'.")
    p.add_argument("--sweep_strategies", type=str, default="mom,stable_lcb,oracle",
                   help="Comma-separated strategies to run in sweep. e.g. 'mom,stable_lcb,flat,oracle'")

    p.add_argument("--out_prefix", type=str, default="bt_run", help="Output prefix for CSV/JSON artifacts.")
    p.add_argument("--data_dir", type=str, default="/Users/oren/Documents/personal/records_data/ticker_date_japan", help="Input data directory.")
    p.add_argument("--resample_sec", type=int, default=1, help="Resample interval in seconds for input data.")
    p.add_argument("--start_date", type=str, default='2025-01-01', help="Start date in YYYY-MM-DD format.")
    p.add_argument("--num_days", type=int, default='7', help="Number of days to process from start_date.")
    p.add_argument("--plot", action="store_true", default=True, help="Generate plot of decisions vs price.")
    p.add_argument("--output_dir", type=str, default="output", help="Output directory for results.")
    return p

def extract_date_from_filename(fname: str) -> Optional[datetime]:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    return None

def prepare_input_data(data_dir: str, resample_sec: int = 1, start_date: str = None, num_days: int = None) -> pd.DataFrame:
    all_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv.gz") or f.endswith(".csv")
    ])
    if not all_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    if start_date and num_days:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=num_days)
        files = []
        for f in all_files:
            fdate = extract_date_from_filename(os.path.basename(f))
            if fdate and start_dt <= fdate < end_dt:
                files.append(f)
    else:
        files = all_files

    if not files:
        raise ValueError(f"No files found for date range starting {start_date} for {num_days} days")

    out_dir = os.path.join(data_dir, "resampled")
    os.makedirs(out_dir, exist_ok=True)

    for fpath in files:
        fname = os.path.basename(fpath).replace(".gz", "")
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue

        try:
            df = pd.read_csv(fpath, compression="gzip" if fpath.endswith(".gz") else None)
        except (EOFError, OSError) as e:
            print(f"Skipping corrupt file: {fpath} ({e})", file=sys.stderr)
            continue

        df["bid"] = pd.to_numeric(df["bid_price"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask_price"], errors="coerce")
        df = df.dropna(subset=["bid", "ask"])
        df = df.loc[(df["bid"] > 0) & (df["ask"] > 0)]
        df = df.loc[df["ask"] >= df["bid"]]
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

        df["_dt"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
        df = df.set_index("_dt").sort_index()
        df_resampled = df[["timestamp", "local_timestamp", "mid"]].resample(f"{resample_sec}s").last().ffill()
        df_resampled = df_resampled.reset_index(drop=True)
        df_resampled.to_csv(out_path, index=False)

    all_combined = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith(".csv") and not f.startswith("_")
    ])
    if start_date and num_days:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=num_days)
        combined_files = [f for f in all_combined if extract_date_from_filename(os.path.basename(f)) and start_dt <= extract_date_from_filename(os.path.basename(f)) < end_dt]
    else:
        combined_files = all_combined
    dfs = [pd.read_csv(f) for f in combined_files]
    combined = pd.concat(dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    combined["_dt"] = pd.to_datetime(combined["timestamp"], unit="us", utc=True)
    combined = combined.set_index("_dt").sort_index()
    combined["p"] = combined["mid"]

    combined_path = os.path.join(out_dir, "_combined_ticker.csv")
    combined.to_csv(combined_path)
    return combined


def main() -> None:
    args = make_argparser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prices_1s = prepare_input_data(args.data_dir, resample_sec=args.resample_sec, start_date=args.start_date, num_days=args.num_days)

    # Parse sweeps
    sweep_tau = parse_list_int(args.sweep_tau)
    sweep_kappa_bps = parse_list_float(args.sweep_kappa_bps)
    sweep_horizon = parse_list_int(args.sweep_horizon)
    sweep_eps_mult = parse_list_float(args.sweep_epsilon_mult)

    do_sweep = any(x is not None for x in [sweep_tau, sweep_kappa_bps, sweep_horizon, sweep_eps_mult])

    if args.sweep_strategies:
        strategies = [s.strip() for s in args.sweep_strategies.split(",") if s.strip()]
    else:
        strategies = [args.strategy]

    # Defaults if not sweeping a dimension
    if sweep_tau is None:
        sweep_tau = [int(args.tau)]
    if sweep_kappa_bps is None:
        sweep_kappa_bps = [float(args.kappa_bps)]
    if sweep_horizon is None:
        sweep_horizon = [int(args.horizon)]
    if sweep_eps_mult is None:
        sweep_eps_mult = [float(args.epsilon_mult)]

    lam = bps_to_return_units(args.lam_bps)

    summary_rows: List[Dict[str, Any]] = []

    for strat in strategies:
        for tau_s in sweep_tau:
            for horizon_s in sweep_horizon:
                for kappa_bps in sweep_kappa_bps:
                    for eps_mult in sweep_eps_mult:
                        win_s = horizon_s // 4
                        kappa = bps_to_return_units(kappa_bps)
                        if kappa_bps <= 0:
                            raise ValueError("kappa_bps must be > 0 for ε-optimality/degeneracy stats. Use positive --kappa_bps/--sweep_kappa_bps.")
                        eps = float(eps_mult) * kappa

                        if strat == "flat":
                            sp = {}
                        elif strat == "mom":
                            sp = {
                                "lookback_s": int(args.mom_lookback),
                                "threshold": bps_to_return_units(args.mom_threshold_bps),
                            }
                        elif strat == "stable_lcb":
                            sp = {
                                "window_s": int(win_s),
                                "beta": float(args.beta),
                            }
                        elif strat == "oracle":
                            sp = {}
                        else:
                            raise ValueError(f"Unknown strategy: {strat}")

                        cfg = BacktestConfig(
                            tau_s=int(tau_s),
                            horizon_s=horizon_s,
                            kappa=kappa,
                            lam=lam,
                            eps=eps,
                            strategy=strat,
                            strategy_params=sp,
                        )

                        # Build run id from config string
                        cfg_str = json.dumps({
                            "tau_s": cfg.tau_s,
                            "horizon_s": cfg.horizon_s,
                            "kappa_bps": float(kappa_bps),
                            "lam_bps": float(args.lam_bps),
                            "eps_mult": float(eps_mult),
                            "strategy": cfg.strategy,
                            "strategy_params": cfg.strategy_params,
                            "price_source": "mid",
                        }, sort_keys=True)

                        run_id = sha1_short(cfg_str)
                        per, met = run_backtest(prices_1s=prices_1s, cfg=cfg)
                        met["run_id"] = run_id
                        met["kappa_bps"] = float(kappa_bps)
                        met["lam_bps"] = float(args.lam_bps)
                        met["eps_mult"] = float(eps_mult)
                        met["window_s"] = int(win_s) if strat == "stable_lcb" else None
                        met["mom_lookback_s"] = int(args.mom_lookback) if strat == "mom" else None
                        met["mom_threshold_bps"] = float(args.mom_threshold_bps) if strat == "mom" else None
                        met["price_source"] = "mid"

                        out_csv = os.path.join(args.output_dir, f"{args.out_prefix}__{run_id}_per_second.csv")
                        out_json = os.path.join(args.output_dir, f"{args.out_prefix}__{run_id}_metrics.json")
                        out_png = os.path.join(args.output_dir, f"{args.out_prefix}__{run_id}_plot.png")
                        per.to_csv(out_csv, index=True)
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(met, f, indent=2, sort_keys=True)
                        if args.plot:
                            plot_decisions_vs_price(per, out_png, title=f"{strat} tau={tau_s}s H={horizon_s}s kappa={kappa_bps}bps")

                        summary_rows.append(met)
                        print(f"[OK] {run_id} strat={strat} tau={tau_s} H={horizon_s} kappa_bps={kappa_bps} eps_mult={eps_mult} W={win_s}")

    # Write sweep summary (even for single run, useful for analysis.py)
    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, f"{args.out_prefix}_sweep_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("\nPrice source: mid")
    print(f"Wrote sweep summary: {summary_csv}")
    print("Tip: run analysis.py on the sweep summary to generate plots.")


if __name__ == "__main__":
    main()
