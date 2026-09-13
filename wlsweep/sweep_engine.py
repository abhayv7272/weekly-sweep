"""
sweep_engine.py — Weekly liquidity-sweep detection for NSE swing trading (buy side).

A "sweep" (a.k.a. stop hunt / turtle soup / liquidity grab) on the weekly timeframe:

   prior weekly swing low = a low lower than the N lows on its left and N on its right
                            (N = swing_strength), OR a fresh low of the lookback window
                            / the 52-week low — both are allowed by design.
   sweep candle           = a COMPLETED weekly candle whose LOW trades below that swing
                            low AND whose CLOSE prints back above that swing low.
   rejection evidence     = a genuine lower wick: close in the upper part of the range,
                            wick dominant vs. body, bar not absurdly wide.

Only candles satisfying all three are flagged, then ranked by a composite Setup Score so
the output is tradable rather than merely "technically true". Nothing here repaints:
the in-progress week is dropped before any signal is computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
@dataclass
class Params:
    # --- structure ---
    swing_strength: int = 2                  # fractal half-width on weekly bars (2 => 5-bar fractal)
    swing_lookback: int = 26                 # weeks to search for swing lows to be swept
    min_swing_age: int = 1                   # swing must be >= this many weeks before the sweep
    max_sweep_bars_ago: int = 1              # 1 => sweep must be on the last completed weekly candle
    include_sweep_on_swing_bar: bool = False  # allow the sweep bar to BE the (unconfirmed) swing bar
    allow_fresh_low: bool = True             # sweeping a fresh low / 52w low is a valid sweep
    require_old_swing: bool = False          # stricter: only confirmed swing lows, no fresh-low sweeps

    # --- sweep geometry ---
    min_close_above_pct: float = 0.0         # close must be >= swept_level * (1 + x)
    min_wick_ratio: float = 0.34             # lower_wick / weekly range
    min_wick_body_mult: float = 1.15         # lower_wick >= k * |body|
    min_close_in_range: float = 0.50         # (close - low) / range
    max_range_of_close: float = 0.12         # reject monster bars (gaps/ex-rights/buybacks)
    min_depth_pct: float = 0.004             # penetration >= 0.4% of level (tick noise filter)
    max_depth_pct: float = 0.16              # but not a violent collapse through the level
    max_depth_atr_mult: float = 3.0          # and not more than this many weekly ATRs

    # --- confirmation ---
    require_green_close: bool = False        # extra strictness: bullish weekly candle
    min_vol_ratio_soft: float = 0.55         # below this, the sweep is penalised, not rejected

    # --- filters (liquidity / tradability) ---
    min_price: float = 5.0
    min_turnover_lakh: float = 75.0          # median daily turnover (Rs lakh) over last 60 sessions
    hard_illiquid_mult: float = 0.35         # < this * min_turnover_lakh => rejected outright
    min_weekly_bars: int = 104
    exclude_locked_weeks: bool = True        # weeks containing a limit/circuit-locked day
    max_locked_weeks_26: int = 1

    # --- score weights (sum ~ 100) ---
    weights: Dict[str, float] = field(default_factory=lambda: {
        "wick": 22.0, "close_position": 16.0, "depth": 12.0, "volume": 10.0,
        "trend": 12.0, "proximity": 10.0, "structure": 10.0, "recency": 8.0,
    })
    min_score: float = 0.0

    def describe(self) -> Dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "weights"}


# --------------------------------------------------------------------------------------
# Weekly resampling
# --------------------------------------------------------------------------------------
def _to_naive_ist(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, format="mixed")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


CIRCUIT_LO, CIRCUIT_HI = 0.095, 0.115      # the 10% band; 20% band derived from it


def resample_weekly(daily: pd.DataFrame, keep_partial: bool = False,
                    today: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Daily bars (DatetimeIndex) -> weekly OHLCV stamped with that week's Monday.

    The unfinished week is dropped unless keep_partial=True, so a signal can never appear
    and disappear mid-candle. Also derives LockedDays (limit/circuit/no-trade days).
    """
    cols = ["Open", "High", "Low", "Close", "Volume", "Days", "Partial", "LockedDays"]
    if daily is None or len(daily) == 0:
        return pd.DataFrame(columns=cols)
    df = daily.copy()
    df = _to_naive_ist(df)
    for c in ("Open", "High", "Low", "Close"):
        if c not in df.columns:
            return pd.DataFrame(columns=cols)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if df.empty:
        return pd.DataFrame(columns=cols)

    dow = df.index.dayofweek                       # Mon=0 .. Sun=6
    week_start = df.index - pd.to_timedelta(dow, unit="D")
    g = df.groupby(week_start, sort=True)
    wk = pd.DataFrame({
        "Open": g["Open"].first(),
        "High": g["High"].max(),
        "Low": g["Low"].min(),
        "Close": g["Close"].last(),
        "Volume": g["Volume"].sum(),
        "Days": g["Open"].size(),
    })
    wk.index = pd.to_datetime(wk.index)
    wk.index.name = "WeekStart"

    last_daily = pd.Timestamp(df.index[-1]).normalize()
    # A week is finished once its Friday session exists (that holds for a half-week ending
    # Thursday too, when Friday was a holiday). If the newest bar is Mon-Thu and "now" is
    # already past it, the week is still forming -> keep it out of the signal tape so the
    # screener cannot repaint mid-candle.
    n = len(wk)
    if n:
        last_days = float(wk["Days"].iloc[-1])
        has_friday = bool((df.index[-1].dayofweek >= 4) and (wk.index[-1] + pd.Timedelta(days=4))
                          in set(pd.DatetimeIndex(df.index).normalize()))
        far_past = (today is not None) and (pd.Timestamp(today).normalize() > last_daily
                                           + pd.Timedelta(days=6))
        forming = (not has_friday) and (last_days < 4.0) and (not far_past)
        wk["Partial"] = forming & (wk.index == wk.index[-1])
    else:
        wk["Partial"] = []
    if not keep_partial:
        wk = wk[~wk["Partial"].to_numpy(dtype=bool)]

    # A week is "locked" only if a day actually *could not trade*: a flat, zero-volume
    # session (halted / no-trade), or a day that pinned at an NSE circuit limit (10%/20%)
    # with a frozen price. A merely large move is NOT a lock - NSE small caps move 5% daily
    # all the time, and treating that as untradeable rejects the whole universe.
    tol = 1e-6
    rng = (df["High"] - df["Low"]).to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    prev = np.concatenate([[np.nan], close[:-1]])
    vol = df["Volume"].to_numpy(dtype=float)
    # a session that traded nobody at all AND did not move (real halt / no-trade band)
    flat = (rng <= np.maximum(tol, 1e-4 * np.abs(close))) & (vol <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.nan_to_num(np.abs(close / prev - 1.0), nan=0.0, posinf=0.0)
    pinned = rng <= np.maximum(tol, 5e-4 * np.abs(close))          # open=high=low=close
    ret = np.nan_to_num(ret, nan=0.0)
    at_limit = (((ret >= CIRCUIT_LO) & (ret <= CIRCUIT_HI)) |
                ((ret >= 2 * CIRCUIT_LO) & (ret <= 2 * CIRCUIT_HI + 0.005)))
    circ = np.asarray(at_limit & pinned, dtype=bool)
    circ = np.nan_to_num(circ.astype(float)).astype(bool)
    lock_daily = pd.Series(np.asarray(flat | circ, dtype=float),
                           index=pd.DatetimeIndex(week_start))
    wk["LockedDays"] = (lock_daily.groupby(level=0).sum().reindex(wk.index).fillna(0.0).to_numpy())
    out = wk.reindex(columns=cols)
    out["LockedDays"] = out["LockedDays"].fillna(0.0)
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=1).mean()


def swing_lows(df: pd.DataFrame, strength: int = 2) -> np.ndarray:
    """Boolean mask: True where LOW is a confirmed fractal swing low.

    Rule: low[i] <= min(low[i-N:i]) and low[i] < min(low[i+1:i+1+N]).
    Strict on the right (so a plateau of equal lows is booked once, on its first bar),
    tolerant on the left. Bars inside the last N bars cannot be confirmed yet — a *fresh
    low* being taken out is handled separately as the "new swing low" case.
    """
    n = int(strength)
    low = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    N = len(low)
    out = np.zeros(N, dtype=bool)
    if n < 1 or N < 2 * n + 1:
        return out
    for i in range(n, N - n):
        cur = low[i]
        if not np.isfinite(cur):
            continue
        left, right = low[i - n:i], low[i + 1:i + 1 + n]
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            continue
        if cur <= left.min() and cur < right.min():
            out[i] = True
    return out


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------
def _ramp(x: float, lo: float, hi: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip((x - lo) / max(hi - lo, EPS), 0.0, 1.0))


def _decay(x: float, half: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(max(0.0, 1.0 - x / max(half, EPS)))


def _band(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid membership: 0 below a, ramps to 1 at b, holds to c, decays to 0 at d."""
    if x is None or not np.isfinite(x):
        return 0.0
    if x <= a:
        return 0.0
    if x < b:
        return float((x - a) / max(b - a, EPS))
    if x <= c:
        return 1.0
    if x >= d:
        return 0.0
    return float((d - x) / max(d - c, EPS))


# --------------------------------------------------------------------------------------
# Core detector
# --------------------------------------------------------------------------------------
@dataclass
class SweepHit:
    sweep_idx: int
    swing_idx: int
    level: float
    date_sweep: pd.Timestamp
    date_swing: pd.Timestamp
    open_: float
    high: float
    low: float
    close: float
    volume: float
    wick_ratio: float
    lower_wick: float
    body: float
    range_: float
    close_in_range: float
    depth_abs: float
    depth_pct: float
    close_above_pct: float
    atr_at_sweep: float
    fresh_low: bool
    is_low_52w: bool
    swing_age_weeks: int
    n_pools: int
    volume_ratio: float
    green: bool
    sma20: float
    sma50: float
    bars_since_sweep: int
    score: float
    drivers: Dict[str, float] = field(default_factory=dict)


def _levels_to_sweep(wk: pd.DataFrame, i: int, p: Params) -> List[Tuple[float, int, bool]]:
    """Candidate liquidity pools for bar i: (level, swing_bar_index, is_fresh_low)."""
    low = wk["Low"].to_numpy(dtype=float)
    j0 = max(0, i - p.swing_lookback)
    out: List[Tuple[float, int, bool]] = []

    if p.swing_strength >= 1:
        mask = swing_lows(wk, p.swing_strength)
        for j in np.flatnonzero(mask):
            j = int(j)
            if j0 <= j <= i - p.min_swing_age:
                out.append((float(low[j]), j, False))
        if p.include_sweep_on_swing_bar:
            # opt-in only: allow the sweep bar to BE the bar that prints the new low
            if np.isfinite(low[i]) and low[i] > 0:
                out.append((float(low[i]), int(i), True))

    def _add(level: float, bar: int, fresh: bool) -> None:
        """Register a candidate pool. A level created BY the test bar itself (i.e. the bar
        simply printing a new low) is only considered when the user opts in — otherwise every
        new-low candle would trivially 'sweep' its own low and the screen would be all noise."""
        if bar == i and not p.include_sweep_on_swing_bar:
            return
        if np.isfinite(level) and level > 0:
            out.append((float(level), int(bar), bool(fresh)))

    # fresh low of the lookback window (a NEW swing low being taken out)
    win = low[j0:i]
    if p.allow_fresh_low and len(win) and np.isfinite(win).all():
        _add(win.min(), int(np.argmin(win)) + j0, True)

    # 52-week low — the most obvious resting liquidity on a weekly chart. Same "new low"
    # family as the running window low, so it is governed by allow_fresh_low, and it is
    # capped by swing_lookback so far-away pools are not silently traded.
    if p.allow_fresh_low:
        s52 = max(0, i - min(52, max(p.swing_lookback, p.min_swing_age + 1)))
        w52 = low[s52:i]
        if len(w52) and np.isfinite(w52).all():
            _add(w52.min(), int(np.argmin(w52)) + s52, True)

    # de-duplicate, keep the earliest swing bar per level
    uniq: Dict[float, Tuple[float, int, bool]] = {}
    for lev, j, fresh in out:
        if not np.isfinite(lev) or lev <= 0:
            continue
        key = round(lev, 6)
        if key not in uniq:
            uniq[key] = (lev, j, fresh)
        else:
            old_lev, old_j, old_fresh = uniq[key]
            # same price: keep the confirmed-swing flag and the oldest bar (one pool, one entry)
            uniq[key] = (old_lev, min(old_j, j), bool(old_fresh) and bool(fresh))
    return list(uniq.values())


def evaluate_bar(wk: pd.DataFrame, i: int, p: Params) -> Optional[SweepHit]:
    """Test one weekly bar for a completed, rejected sweep of a prior swing low."""
    if i < 1 or i >= len(wk):
        return None
    lo = float(wk["Low"].iloc[i]); hi = float(wk["High"].iloc[i])
    cl = float(wk["Close"].iloc[i]); op = float(wk["Open"].iloc[i])
    if not all(np.isfinite(x) for x in (lo, hi, cl, op)) or cl <= 0 or lo <= 0:
        return None
    rng = hi - lo
    if rng <= 0:
        return None
    body = abs(cl - op)
    lower_wick = min(op, cl) - lo
    wick_ratio = lower_wick / rng
    close_in_range = (cl - lo) / rng

    # --- rejection evidence -------------------------------------------------------------
    if wick_ratio < p.min_wick_ratio:
        return None
    if lower_wick < p.min_wick_body_mult * body:
        return None
    if close_in_range < p.min_close_in_range:
        return None
    if rng / cl > p.max_range_of_close:
        return None
    if p.require_green_close and cl <= op:
        return None

    # --- which pool did it take? -------------------------------------------------------
    a = float(atr(wk, 14).iloc[i])
    if not np.isfinite(a) or a <= 0:
        a = rng
    best: Optional[Tuple[float, int, bool, float]] = None
    n_pools = 0
    for lev, j, fresh in _levels_to_sweep(wk, i, p):
        depth = lev - lo
        if depth <= 0:
            continue                                   # level not actually swept
        if depth < p.min_depth_pct * lev:              # tick-noise floor
            continue                                   # tick noise, not a sweep
        if depth > p.max_depth_pct * lev:
            continue                                   # collapsed through the level
        if depth > p.max_depth_atr_mult * a:
            continue
        if cl <= lev * (1.0 + p.min_close_above_pct):
            continue                                   # must CLOSE back above the level
        if fresh and not p.allow_fresh_low:
            continue
        if p.require_old_swing and fresh:
            continue
        n_pools += 1
        # the deepest pool taken out is the one that matters (one candle clearing several
        # stacked lows = a much bigger liquidity grab than nudging the nearest one)
        if best is None or lev < best[0]:
            best = (lev, j, fresh, depth)
    if best is None:
        return None
    lev, j, fresh, depth = best

    s52w = max(0, i - 52)
    w52 = wk["Low"].to_numpy(dtype=float)[s52w:i]
    is52 = bool(len(w52) and np.isfinite(w52).all() and lev <= float(w52.min()) * (1 + 1e-6))

    vol = float(wk["Volume"].iloc[i]) if "Volume" in wk.columns else np.nan
    vseq = pd.to_numeric(wk["Volume"], errors="coerce").to_numpy(dtype=float)
    vma = np.nanmean(vseq[max(0, i - 20):i]) if i > 0 else np.nan
    vol_ratio = float(vol / vma) if np.isfinite(vma) and vma > 0 and np.isfinite(vol) else 1.0

    sma20 = float(wk["Close"].rolling(20, min_periods=8).mean().iloc[i])
    sma50 = float(wk["Close"].rolling(50, min_periods=25).mean().iloc[i])
    drivers = _score_drivers(wk, i, dict(
        low=lo, close=cl, level=lev, depth=depth, atr=a, wick_ratio=wick_ratio,
        close_in_range=close_in_range, green=(cl > op), vol_ratio=vol_ratio,
        sma20=sma20, sma50=sma50, bars_since_sweep=int(len(wk) - 1 - i),
        fresh=fresh, is52=is52, swing_age=int(i - j),
    ), p)
    w = p.weights
    score = float(np.clip(sum(w.get(k, 0.0) * v for k, v in drivers.items()), 0.0, 100.0))

    return SweepHit(
        sweep_idx=int(i), swing_idx=int(j), level=float(lev),
        date_sweep=pd.Timestamp(wk.index[i]), date_swing=pd.Timestamp(wk.index[j]),
        open_=op, high=hi, low=lo, close=cl, volume=vol if np.isfinite(vol) else np.nan,
        wick_ratio=float(wick_ratio), lower_wick=float(lower_wick), body=float(body),
        range_=float(rng), close_in_range=float(close_in_range),
        depth_abs=float(depth), depth_pct=float(depth / lev),
        close_above_pct=float(cl / lev - 1.0), atr_at_sweep=float(a),
        fresh_low=bool(fresh), is_low_52w=is52, swing_age_weeks=int(i - j), n_pools=int(n_pools),
        volume_ratio=float(vol_ratio), green=bool(cl > op),
        sma20=sma20 if np.isfinite(sma20) else np.nan,
        sma50=sma50 if np.isfinite(sma50) else np.nan,
        bars_since_sweep=int(len(wk) - 1 - i), score=score,
        drivers={k: round(float(v), 3) for k, v in drivers.items()},
    )


def _score_drivers(wk: pd.DataFrame, i: int, f: dict, p: Params) -> Dict[str, float]:
    d: Dict[str, float] = {}
    d["wick"] = _ramp(f["wick_ratio"], p.min_wick_ratio, 0.75)
    d["close_position"] = _ramp(f["close_in_range"], p.min_close_in_range, 0.95) * (1.15 if f["green"] else 0.85)
    depth_pct = f["depth"] / max(f["level"], EPS)
    d["depth"] = _band(depth_pct, 0.002, 0.012, 0.075, 0.16)
    vr = f["vol_ratio"]
    if vr >= 1.0:
        d["volume"] = _ramp(vr, 0.9, 2.6)
    else:
        d["volume"] = max(0.0, _ramp(vr, p.min_vol_ratio_soft, 1.0) * 0.7)
    c, s20, s50 = f["close"], f["sma20"], f["sma50"]
    tr = 0.0
    if np.isfinite(s20) and s20 > 0:
        tr += 0.6 if c >= s20 else (0.3 if c >= s20 * 0.985 else 0.0)
    if np.isfinite(s50) and s50 > 0:
        tr += 0.4 if s20 >= s50 * 0.98 else 0.0
    d["trend"] = min(1.0, tr)
    risk_pct = (c - f["low"]) / max(c, EPS)
    d["proximity"] = _band(risk_pct, 0.005, 0.02, 0.10, 0.17)
    st = 0.90 if f["is52"] else (0.72 if not f["fresh"] else 0.62)
    lo52 = float(np.nanmin(wk["Low"].to_numpy(dtype=float)[max(0, i - 52):i + 1])) if i > 0 else np.nan
    if np.isfinite(lo52) and lo52 > 0:
        st *= 1.0 - 0.55 * _ramp(c / lo52 - 1.0, 0.35, 1.10)      # don't chase far above the base
    d["structure"] = float(np.clip(st, 0.0, 1.0))
    d["recency"] = 1.0 if f["bars_since_sweep"] == 0 else max(0.0, 1.0 - 0.28 * f["bars_since_sweep"])
    return d


def find_sweeps(wk: pd.DataFrame, p: Params) -> List[SweepHit]:
    """Scan the most recent `max_sweep_bars_ago` completed weekly bars; return hits best-first."""
    n = len(wk)
    if n < max(p.min_weekly_bars, 2 * p.swing_strength + 3):
        return []
    last_i = n - 1
    span = max(1, int(p.max_sweep_bars_ago))
    first_i = max(p.swing_strength + 1, last_i - span + 1)
    hits = [h for h in (evaluate_bar(wk, i, p) for i in range(first_i, last_i + 1)) if h is not None]
    if not hits:
        return []
    hits = [h for h in hits if h.score >= p.min_score]
    # newest first for ties: a 3-week-old sweep must never outrank today's
    return sorted(hits, key=lambda h: (-round(h.score, 4), h.bars_since_sweep))


def sweep_in_window(wk: pd.DataFrame, p: Params) -> Tuple[Optional[SweepHit], Optional[dict]]:
    """
    Primary entry point of the screener. Returns (hit, near_miss_dict).

    Honours `max_sweep_bars_ago`: 1 = "the sweep must be on the newest closed week" (the
    strict, tradable-now reading); N = "a sweep completed within the last N weeks".
    A hit anywhere in the window suppresses near-miss reporting for that same window, so the
    watchlist never duplicates a row that already qualified.
    """
    n = len(wk)
    if n == 0:
        return None, None
    hits = find_sweeps(wk, p)
    if hits:
        return hits[0], None                       # already newest-first, best-score-first
    span = max(1, int(p.max_sweep_bars_ago))
    for i in range(n - 1, max(n - 1 - span, p.swing_strength) - 1, -1):
        nm = near_miss(wk, p, i)
        if nm is not None:
            return None, nm
    return None, None


# kept for the tests / older calls: window of exactly one bar == "the last bar"
def sweep_on_last_bar(wk: pd.DataFrame, p: Params, window: int = 1
                      ) -> Tuple[Optional[SweepHit], Optional[dict]]:
    if window != max(1, int(p.max_sweep_bars_ago)):
        p = Params(**{**p.__dict__, "max_sweep_bars_ago": int(window)})
    return sweep_in_window(wk, p)


def near_miss(wk: pd.DataFrame, p: Params, i: Optional[int] = None) -> Optional[dict]:
    """
    Bars that DID sweep a level and closed above it but failed a quality gate.
    Reported as a watchlist, and as proof the detector is not blind.
    """
    n = len(wk)
    i = (n - 1) if i is None else i
    if n < 3 or i < 1:
        return None
    lo = float(wk["Low"].iloc[i]); hi = float(wk["High"].iloc[i])
    cl = float(wk["Close"].iloc[i]); op = float(wk["Open"].iloc[i])
    if not all(np.isfinite(x) for x in (lo, hi, cl, op)) or cl <= 0:
        return None
    rng = hi - lo
    if rng <= 0:
        return None
    swept = [lev for lev, j, fresh in _levels_to_sweep(wk, i, p)
             if lo < lev and cl > lev * (1.0 + p.min_close_above_pct)]
    if not swept:
        return None
    level = min(swept)
    lower_wick = min(op, cl) - lo
    reasons = []
    if lower_wick / rng < p.min_wick_ratio:
        reasons.append("wick too small")
    if lower_wick < p.min_wick_body_mult * abs(cl - op):
        reasons.append("body larger than wick")
    if (cl - lo) / rng < p.min_close_in_range:
        reasons.append("close weak / not in upper half")
    if rng / cl > p.max_range_of_close:
        reasons.append("weekly range too wide")
    if p.require_green_close and cl <= op:
        reasons.append("red candle")
    if not reasons:
        return None
    return {
        "Week": str(pd.Timestamp(wk.index[i]).date()), "Close": round(cl, 2),
        "SweptLevel": round(level, 2), "SweepLow": round(lo, 2),
        "WickRatio%": round(lower_wick / rng * 100, 1),
        "CloseInRange%": round((cl - lo) / rng * 100, 1),
        "FailReason": ", ".join(reasons),
    }


# --------------------------------------------------------------------------------------
# Universe filters + master screen
# --------------------------------------------------------------------------------------
def turnover_lakh(df: pd.DataFrame, n: int = 60) -> float:
    """Median daily rupee turnover over the last `n` sessions, in lakh (1 lakh = 1e5)."""
    if df is None or len(df) == 0:
        return np.nan
    v = pd.to_numeric(df["Volume"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    m = np.nanmedian(v[-n:] * c[-n:]) / 1e5
    return float(m) if np.isfinite(m) else np.nan


def locked_weeks(wk: pd.DataFrame, n: int = 26) -> int:
    if wk is None or "LockedDays" not in wk.columns or len(wk) == 0:
        return 0
    arr = pd.to_numeric(wk["LockedDays"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return int((arr[-n:] >= 1).sum())


def screen_all(data: Dict[str, pd.DataFrame], p: Params,
                universe: Optional[pd.DataFrame] = None,
                keep_near_miss: bool = True,
                progress_cb: Optional[Callable[[int, int], None]] = None,
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Screen every cached symbol.

    Returns
    -------
    hits       : confirmed setups (sorted by score, newest first)
    near_miss  : candles that swept + closed above but failed a quality gate
    diagnostics: one row per screened symbol. status is
                 hit | near_miss | no_setup | insufficient_weekly_bars | price_below_min
                 | illiquid | circuit_locked | error:*, i.e. nothing is left unexplained
    """
    hit_rows, miss_rows, diag_rows = [], [], []
    uni = None
    if universe is not None and len(universe) and "Symbol" in universe.columns:
        uni = universe.drop_duplicates("Symbol").set_index("Symbol")

    for k, (sym, daily) in enumerate(data.items()):
        if progress_cb is not None and k % 50 == 0:
            progress_cb(k, len(data))
        # "ok" here read as "passed every filter" in the status table; "hit" is unambiguous
        rec: Dict[str, object] = {"Symbol": sym, "status": "hit"}
        try:
            wk = resample_weekly(daily)
        except Exception as exc:  # defensive: never let one symbol kill the run
            diag_rows.append({**rec, "status": f"error:{type(exc).__name__}"})
            continue
        rec["WeeklyBars"] = int(len(wk))
        if len(wk) < p.min_weekly_bars:
            diag_rows.append({**rec, "status": "insufficient_weekly_bars"})
            continue
        cl = float(wk["Close"].iloc[-1])
        rec["Close"] = round(cl, 2)
        rec["LastWeek"] = str(pd.Timestamp(wk.index[-1]).date())
        rec["TurnoverLakh"] = round(turnover_lakh(daily), 2) if np.isfinite(turnover_lakh(daily)) else np.nan
        rec["LockedWeeks26"] = locked_weeks(wk)

        if cl < p.min_price:
            diag_rows.append({**rec, "status": "price_below_min"}); continue
        to = rec["TurnoverLakh"]
        if np.isfinite(to) and to < p.min_turnover_lakh * p.hard_illiquid_mult:
            diag_rows.append({**rec, "status": "illiquid"}); continue
        if p.exclude_locked_weeks and rec["LockedWeeks26"] > p.max_locked_weeks_26:
            diag_rows.append({**rec, "status": "circuit_locked"}); continue

        hit, miss = sweep_in_window(wk, p)
        if hit is not None:
            hit_rows.append(_row_from_hit(sym, hit, wk, uni, rec))
        elif keep_near_miss and miss is not None:
            miss_rows.append({"Symbol": sym, **miss})
        if hit is None and miss is not None:
            rec["status"] = "near_miss"
        elif hit is None:
            rec["status"] = "no_setup"
        else:
            rec["status"] = "hit"
        diag_rows.append(rec)

    hits = pd.DataFrame(hit_rows)
    if not hits.empty:
        hits = hits.sort_values(["Score", "BarsSinceSweep"], ascending=[False, True]).reset_index(drop=True)
        hits.insert(0, "Rank", np.arange(1, len(hits) + 1))
    return hits, pd.DataFrame(miss_rows), pd.DataFrame(diag_rows)


def _row_from_hit(sym: str, h: SweepHit, wk: pd.DataFrame, uni: Optional[pd.DataFrame],
                  rec: Dict[str, object]) -> dict:
    i = h.sweep_idx
    stop = h.low * 0.995
    risk = h.close - stop
    seg = pd.to_numeric(wk["High"].iloc[max(0, i - 26): i + 1], errors="coerce").to_numpy(dtype=float)
    seg = seg[np.isfinite(seg)]
    resist = float(seg.max()) if len(seg) else np.nan
    row = {
        "Symbol": sym,
        "Company": (str(uni.at[sym, "Company"]) if uni is not None and "Company" in uni.columns
                    and sym in uni.index else ""),
        "Close": round(h.close, 2),
        "SweepWeek": str(pd.Timestamp(h.date_sweep).date()),
        "SweptSwingLow": round(h.level, 2),
        "SweepLow": round(h.low, 2),
        "SwingLowMade": str(pd.Timestamp(h.date_swing).date()),
        "SwingAgeW": int(h.swing_age_weeks),
        "PoolsTaken": int(h.n_pools),
        "ClosedAboveBy%": round(h.close_above_pct * 100, 2),
        "SweepDepth%": round(h.depth_pct * 100, 2),
        "WickRatio%": round(h.wick_ratio * 100, 1),
        "CloseInRange%": round(h.close_in_range * 100, 1),
        "Green": bool(h.green),
        "SweepType": ("52-week-low sweep" if h.is_low_52w
                      else "new-low sweep" if h.fresh_low else "prior swing-low sweep"),
        "LowPoolRef": (f"lowest of {h.n_pools} pools taken" if h.n_pools > 1 else ""),
        "Is52wLow": bool(h.is_low_52w),
        "VolRatio": round(h.volume_ratio, 2),
        "ATR(14w)": round(h.atr_at_sweep, 2),
        "SMA20w": round(h.sma20, 2) if np.isfinite(h.sma20) else np.nan,
        "vsSMA20w%": round((h.close / h.sma20 - 1) * 100, 2) if np.isfinite(h.sma20) and h.sma20 else np.nan,
        "BarsSinceSweep": int(len(wk) - 1 - i),
        "Risk%": round(risk / h.close * 100, 2),
        "Stop": round(stop, 2),
        "Target2R": round(h.close + 2 * risk, 2),
        "Target3R": round(h.close + 3 * risk, 2),
        "Resistance26w": round(resist, 2) if np.isfinite(resist) else np.nan,
        "UpsideToRes%": round((resist / h.close - 1) * 100, 2) if np.isfinite(resist) else np.nan,
        "TurnoverLakh": rec.get("TurnoverLakh", np.nan),
        "Score": round(h.score, 1),
    }
    for k, v in h.drivers.items():
        row[f"S_{k}"] = v
    return row


# --------------------------------------------------------------------------------------
# Top-N ranking of the universe
# --------------------------------------------------------------------------------------
def rank_top(data: Dict[str, pd.DataFrame], quotes: Optional[pd.DataFrame] = None,
             top_n: int = 1000, index_tiers: Optional[pd.DataFrame] = None,
             prefer_index_tier: bool = False) -> pd.DataFrame:
    """
    Decide the "top N" of the universe.

    Ranking key, in order of preference:
      1. real market cap      - only when the optional quote endpoint cooperated
      2. index tier, turnover - when prefer_index_tier=True (mirrors the official Nifty
                                50/100/200/500 ordering)
      3. turnover             - default; always available, and it is the honest ordering
                                for a *liquidity* screener
    Stocks outside every index still get a fair shot at the top 1000 in modes 1 and 3.

    `index_tiers` : DataFrame[Symbol, IndexTier, Industry] from nse_data.load_index_membership
    Returns columns: Rank, Symbol, Close, TurnoverLakh, IndexTier, Industry, rank_basis
    """
    rows = []
    for sym, df in data.items():
        c = pd.to_numeric(df.get("Close"), errors="coerce").to_numpy(dtype=float)
        px = float(c[-1]) if len(c) and np.isfinite(c[-1]) else np.nan
        rows.append({"Symbol": sym, "Close": px, "TurnoverLakh": turnover_lakh(df)})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return tab
    tab["RankLakh"] = tab["TurnoverLakh"].rank(ascending=False, method="min")

    if index_tiers is not None and len(index_tiers):
        it = index_tiers.drop_duplicates("Symbol").set_index("Symbol")
        tab["IndexTier"] = tab["Symbol"].map(it["IndexTier"]) if "IndexTier" in it.columns else None
        tab["Industry"] = tab["Symbol"].map(it["Industry"]) if "Industry" in it.columns else None
        order = {k: n for n, k in enumerate(
            ["Nifty50", "Nifty100", "Nifty200", "Nifty500", "NiftyMidcap150", "NiftySmallcap250"])}
        tab["_tier"] = tab["IndexTier"].map(lambda x: order.get(x, 99)).fillna(99).astype(int)
    else:
        tab["IndexTier"] = None
        tab["Industry"] = None
        tab["_tier"] = 99

    basis = "index_tier+turnover"
    if quotes is not None and len(quotes) and "MarketCap" in quotes.columns:
        tab = tab.merge(quotes[["Symbol", "MarketCap"]].drop_duplicates("Symbol"),
                        on="Symbol", how="left")
        if float(tab["MarketCap"].notna().mean() or 0) > 0.6:
            basis = "market_cap"
    else:
        tab["MarketCap"] = np.nan

    if basis == "market_cap":
        keys, asc = ["_mcap", "_tier", "Symbol"], [False, True, True]
    elif prefer_index_tier:
        keys, asc = ["_tier", "RankLakh", "Symbol"], [True, True, True]
    else:
        keys, asc = ["RankLakh", "_tier", "Symbol"], [True, True, True]
    tab["_mcap"] = tab["MarketCap"]
    tab["RankLakh"] = tab["RankLakh"].fillna(1e12)
    tab = tab.sort_values(keys, ascending=asc, na_position="last")
    tab = tab.head(int(top_n)).drop(columns=["_mcap", "_tier", "RankLakh"],
                                    errors="ignore").reset_index(drop=True)
    tab.insert(0, "Rank", np.arange(1, len(tab) + 1))
    tab["rank_basis"] = basis
    return tab


# --------------------------------------------------------------------------------------
# Backtest of the same rule (parameter sanity, not a promise)
# --------------------------------------------------------------------------------------
def backtest(data: Dict[str, pd.DataFrame], p: Params, horizon_weeks: int = 8,
             r_mult: float = 2.0, cost_pct: float = 0.15,
             max_symbols: Optional[int] = None) -> pd.DataFrame:
    """
    Same geometry rule, replayed on the whole history of each symbol:
    entry at the open of the week AFTER the sweep, stop = sweep low -0.5%,
    target = entry + r_mult*risk, otherwise exit at the close of week `horizon_weeks`.
    """
    rows = []
    items = list(data.items())
    if max_symbols:
        items = items[:max_symbols]
    for sym, daily in items:
        wk = resample_weekly(daily)
        if len(wk) < p.min_weekly_bars:
            continue
        lo_a = wk["Low"].to_numpy(float); hi_a = wk["High"].to_numpy(float)
        op_a = wk["Open"].to_numpy(float); cl_a = wk["Close"].to_numpy(float)
        atr_a = atr(wk, 14).to_numpy(float)
        sw = np.flatnonzero(swing_lows(wk, p.swing_strength))
        for i in range(p.swing_strength + 1, len(wk) - 1):
            rng = hi_a[i] - lo_a[i]
            if rng <= 0 or not np.isfinite(cl_a[i]) or cl_a[i] <= 0:
                continue
            lw = min(op_a[i], cl_a[i]) - lo_a[i]
            if lw / rng < p.min_wick_ratio or lw < p.min_wick_body_mult * abs(cl_a[i] - op_a[i]):
                continue
            if (cl_a[i] - lo_a[i]) / rng < p.min_close_in_range or rng / cl_a[i] > p.max_range_of_close:
                continue
            j0 = max(0, i - p.swing_lookback)
            cands = [float(wk["Low"].iloc[j]) for j in sw[(sw >= j0) & (sw <= i - p.min_swing_age)]]
            win = lo_a[j0:i]
            if len(win) and np.isfinite(win).all() and p.allow_fresh_low:
                cands.append(float(win.min()))
            a = atr_a[i] if np.isfinite(atr_a[i]) and atr_a[i] > 0 else rng
            ok = []
            for lev in cands:
                if not np.isfinite(lev) or lev <= 0:
                    continue
                depth = lev - lo_a[i]
                if depth <= 0 or cl_a[i] <= lev * (1 + p.min_close_above_pct):
                    continue
                if depth < p.min_depth_pct * lev or depth > p.max_depth_pct * lev:
                    continue
                if depth > p.max_depth_atr_mult * a:
                    continue
                ok.append(lev)
            if not ok:
                continue
            level = max(ok)
            entry = op_a[i + 1]
            stop = lo_a[i] * 0.995
            risk = entry - stop
            if not np.isfinite(entry) or entry <= 0 or risk <= 0:
                continue
            target = entry + r_mult * risk
            j_end = min(len(wk) - 1, i + 1 + max(1, int(horizon_weeks)))
            out, brk = cl_a[j_end], "time"
            for j in range(i + 1, j_end + 1):
                if lo_a[j] <= stop:
                    out, brk = stop, "stop"; break
                if hi_a[j] >= target:
                    out, brk = target, "target"; break
            ret = (out / entry - 1) * 100 - cost_pct
            rows.append({"Symbol": sym, "SweepWeek": str(pd.Timestamp(wk.index[i]).date()),
                         "SweptLevel": round(level, 2), "Entry": round(entry, 2),
                         "Stop": round(stop, 2), "Exit": round(out, 2), "Outcome": brk,
                         "Return%": round(ret, 2), "R": round((out - entry) / risk, 2),
                         "BarsToExit": int(j_end - i - 1)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Symbol", "SweepWeek"]).reset_index(drop=True)
    return df


def summarize_backtest(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0}
    r = trades["Return%"].to_numpy(float)
    r = r[np.isfinite(r)]
    if not len(r):
        return {"trades": 0}
    pos, neg = r[r > 0], r[r < 0]
    return {
        "trades": int(len(r)),
        "win_rate_%": round(100 * float((r > 0).mean()), 1),
        "avg_return_%": round(float(np.mean(r)), 2),
        "median_return_%": round(float(np.median(r)), 2),
        "avg_R": round(float(trades["R"].mean()), 2),
        "hit_target_%": round(100 * float((trades["Outcome"] == "target").mean()), 1),
        "hit_stop_%": round(100 * float((trades["Outcome"] == "stop").mean()), 1),
        "expired_%": round(100 * float((trades["Outcome"] == "time").mean()), 1),
        "profit_factor": round(float(min(99.0, np.sum(pos) / max(EPS, abs(np.sum(neg))))), 2),
        "p5_%": round(float(np.percentile(r, 5)), 2),
        "p95_%": round(float(np.percentile(r, 95)), 2),
    }


# --------------------------------------------------------------------------------------
# Chart payload (consumed by the notebook's plotting cell)
# --------------------------------------------------------------------------------------
def chart_frames(daily: pd.DataFrame, p: Params, weeks_to_show: int = 60
                 ) -> Tuple[pd.DataFrame, dict]:
    """Weekly tail for candlestick plotting + the swing/sweep annotations to overlay."""
    wk = resample_weekly(daily)
    if wk.empty:
        return wk, {"hit": None, "swings": [], "first_index": 0}
    hits = find_sweeps(wk, Params(**{**asdict(p)}))
    start = max(0, len(wk) - weeks_to_show)
    tail = wk.iloc[start:].copy()
    info: dict = {"hit": None, "swings": [], "first_index": int(start), "levels": []}
    mask = swing_lows(wk, p.swing_strength)
    for j in np.flatnonzero(mask):
        if j >= start:
            info["swings"].append({"pos": int(j - start), "date": str(pd.Timestamp(wk.index[j]).date()),
                                   "low": float(wk["Low"].iloc[j])})
    # the level(s) currently worth sweeping
    i = len(wk) - 1
    for lev, j, fresh in _levels_to_sweep(wk, i, p):
        info["levels"].append({"level": round(lev, 2), "date": str(pd.Timestamp(wk.index[j]).date()),
                               "fresh": bool(fresh)})
    cand = [h for h in hits if h.sweep_idx == i] or hits
    if cand:
        h = cand[0]
        d = asdict(h)
        d["sweep_pos"] = int(h.sweep_idx - start)
        d.pop("date_sweep", None); d.pop("date_swing", None)
        info["hit"] = d
    return tail, info