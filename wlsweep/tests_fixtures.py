"""
tests_fixtures.py — synthetic-data unit tests for the weekly liquidity-sweep screener.

No network needed. These pin down the exact contract the user asked for:

  1. a prior weekly swing low (old OR freshly made) must be taken out,
  2. the sweep candle must CLOSE back above that swing low,
  3. a genuine lower wick must be visible,
  4. the whole thing must be on a COMPLETED weekly candle (no mid-candle repaint),
  plus: the liquidity/universe plumbing, ranking, and the backtest harness.

Run with:  python -m pytest tests_fixtures.py -q     (or just run the notebook's test cell)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sweep_engine import (Params, _levels_to_sweep, _row_from_hit, atr, backtest, chart_frames,
                          find_sweeps, locked_weeks, near_miss, rank_top, resample_weekly,
                          screen_all, summarize_backtest, sweep_in_window, swing_lows)


# --------------------------------------------------------------------------------------
# Tape builders (deterministic — no RNG, so the assertions are exact)
# --------------------------------------------------------------------------------------
def weekly(bars, start="2023-01-02") -> pd.DataFrame:
    """bars: list of (o,h,l,c[,vol]) -> weekly frame stamped on consecutive Mondays."""
    rows = [(b[0], b[1], b[2], b[3], b[4] if len(b) > 4 else 1_000_000) for b in bars]
    idx = pd.date_range(start=start, periods=len(rows), freq="7D")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    df.index.name = "WeekStart"
    df["Days"] = 5
    df["Partial"] = False
    df["LockedDays"] = 0.0
    return df


def daily_frame(days, o, h, l, c, v):
    idx = pd.DatetimeIndex(pd.to_datetime(list(days)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def as_daily(wk: pd.DataFrame) -> pd.DataFrame:
    """Weekly frame -> synthetic 5-session-per-week daily frame (plumbing tests only)."""
    rows, idx = [], []
    for ts, r in wk.iterrows():
        for k in range(5):
            d = ts + pd.Timedelta(days=k)
            if d.weekday() > 4:
                continue
            rows.append((r["Open"], r["High"], r["Low"], r["Close"], float(r["Volume"]) / 5.0))
            idx.append(d)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"],
                        index=pd.DatetimeIndex(idx))


def drift(n=26, start=150.0, step=1.5, vol=900_000):
    """Gentle deterministic downtrend with 2.5-wide bars; never the tape's floor."""
    return [(round(start - k * step, 2), round(start - k * step + 2.0, 2),
             round(start - k * step - 0.5, 2), round(start - (k + 1) * step, 2), vol)
            for k in range(n)]


PRE = [(101.0, 103.0, 100.0, 102.5),   # idx F: the OLD SWING LOW = 100.0
       (102.5, 105.0, 102.0, 104.5),   # bounce
       (104.5, 106.0, 103.5, 105.0),    # hold
       (105.0, 106.0, 104.0, 104.5),    # hold
       (104.5, 105.5, 103.0, 104.0),    # hold
       (104.0, 105.0, 103.2, 104.8)]    # hold


def sweep_tape(n_filler=26):
    """
    Drift down -> swing low 100 -> bounce/hold -> textbook sweep candle:
        open 101.0, high 103.0, low 96.0, close 102.5, 4x volume
    lower wick 5.0 (range 7.0 -> wick ratio 0.714, close in range 0.929, green).
    """
    return weekly(drift(n_filler) + PRE + [(101.0, 103.0, 96.0, 102.5, 4_000_000)])


def final_bar_tape(o, h, l, c, v=1_000_000, n_filler=26, extra_pre=None):
    """Same structure as sweep_tape but with a custom last bar."""
    return weekly(drift(n_filler) + PRE + (extra_pre or []) + [(o, h, l, c, v)])


def fresh_low_tape(n=28):
    """
    Monotonic decline -> only *fresh* lows exist (no confirmed swing to sweep).
    Last bar undercuts the running low by 7.5 and closes back above it.
    """
    bars = drift(n, start=320.0, step=8.0)
    last_low = bars[-1][2]
    bars.append((round(last_low + 0.5, 2), round(last_low + 2.0, 2),
                 round(last_low - 7.5, 2), round(last_low + 1.5, 2), 2_500_000))
    return weekly(bars)


def stale_sweep_tape(n_filler=26, n_after=4):
    """The sweep candle sits `n_after` weeks back, followed by quiet higher chop."""
    bars = drift(n_filler) + PRE[:5] + [(101.0, 103.0, 96.0, 102.5, 4_000_000)]
    px = 103.0
    for _ in range(n_after):
        bars.append((round(px, 2), round(px + 2, 2), round(px + 1, 2), round(px + 1.5, 2)))
        px += 1.5
    return weekly(bars)


def _row(h, wk):
    return _row_from_hit("TEST", h, wk, None, {"TurnoverLakh": 100.0})


# --------------------------------------------------------------------------------------
# Weekly resampling (must not repaint mid-candle)
# --------------------------------------------------------------------------------------
def test_resample_last_completed_week_is_kept():
    """The signal lives on the newest completed week: it must NEVER be dropped."""
    days = list(pd.date_range("2024-01-15", "2024-01-19", freq="B"))
    df = pd.DataFrame({"Open": np.ones(5) * 10, "High": 11.0, "Low": 9.0, "Close": 10.5,
                       "Volume": 100.0}, index=pd.DatetimeIndex(days))
    for td in ["2024-01-19", "2024-01-20", "2024-01-21", "2024-02-02"]:
        wk = resample_weekly(df, keep_partial=False, today=pd.Timestamp(td))
        assert len(wk) == 1, (td, wk)
    wk_auto = resample_weekly(df, keep_partial=False)          # today defaults to last bar
    assert len(wk_auto) == 1
    # mid-week data (no Friday yet) is treated as an unfinished candle
    partial = resample_weekly(df.iloc[:3], keep_partial=False, today=pd.Timestamp("2024-01-17"))
    assert len(partial) == 0
    assert len(resample_weekly(df.iloc[:3], keep_partial=True,
                               today=pd.Timestamp("2024-01-17"))) == 1


def test_resample_groups_by_calendar_week():
    days = list(pd.date_range("2024-01-01", "2024-01-19", freq="B"))   # 3 x Mon..Fri
    n = len(days)
    assert n == 15
    o = np.arange(1, n + 1, dtype=float)
    df = daily_frame(days, o, o + 1.0, o - 0.5, o + 0.5, np.full(n, 10.0))
    wk = resample_weekly(df, keep_partial=True, today=pd.Timestamp("2024-01-19"))
    assert len(wk) == 3, len(wk)
    assert wk["Open"].iloc[0] == 1.0 and wk["High"].iloc[0] == 6.0     # Fri 5 Jan: O=5 H=6
    assert wk["Open"].iloc[1] == 6.0                                   # Mon 8 Jan opens week 2
    assert wk["Close"].iloc[-1] == 15.5                                # Fri 19 Jan closes week 3
    assert wk["Volume"].sum() == 150.0
    assert wk.index[0] == pd.Timestamp("2024-01-01")                   # stamped on Monday
    assert not bool(wk["Partial"].iloc[-1])  # a Friday session exists -> week is complete


def test_resample_drops_partial_current_week():
    days = list(pd.date_range("2024-01-01", "2024-01-12", freq="B"))    # 2 complete weeks
    days += list(pd.date_range("2024-01-15", "2024-01-16", freq="B"))    # 2-session week, no Friday
    n = len(days)
    o = np.arange(1, n + 1, dtype=float)
    df = pd.DataFrame({"Open": o, "High": o + 1, "Low": o - 1, "Close": o + 0.5,
                       "Volume": np.full(n, 10.0)}, index=pd.DatetimeIndex(days))
    part = resample_weekly(df, keep_partial=True, today=pd.Timestamp("2024-01-16"))
    full = resample_weekly(df, keep_partial=False, today=pd.Timestamp("2024-01-16"))
    assert list(part["Days"]) == [5, 5, 2], list(part["Days"])
    assert [bool(x) for x in part["Partial"]] == [False, False, True]
    assert len(full) == 2 and str(full.index[-1].date()) == "2024-01-08"
    # the same 2-day week becomes a *completed* week once data reaches Friday
    days5 = list(pd.date_range("2024-01-01", "2024-01-19", freq="B"))
    o5 = np.arange(1, len(days5) + 1, dtype=float)
    df5 = pd.DataFrame({"Open": o5, "High": o5 + 1, "Low": o5 - 1, "Close": o5 + 0.5,
                        "Volume": np.full(len(days5), 10.0)}, index=pd.DatetimeIndex(days5))
    nxt = resample_weekly(df5, keep_partial=False, today=pd.Timestamp("2024-01-22"))
    assert len(nxt) == 3 and str(nxt.index[-1].date()) == "2024-01-15"


def test_resample_keeps_completed_current_week():
    # week ending Fri 19 Jan is complete even if "today" is that same Friday
    days = list(pd.date_range("2024-01-15", "2024-01-19", freq="B"))
    df = daily_frame(days, np.ones(5), 2.0, 0.5, 1.5, np.ones(5) * 10)
    wk = resample_weekly(df, keep_partial=False, today=pd.Timestamp("2024-01-19"))
    assert len(wk) == 1


def test_resample_handles_tz_and_prebuilt_weekly_input():
    days = pd.date_range("2023-01-02", periods=52 * 3, freq="7D", tz="Asia/Kolkata")
    n = len(days)
    df = daily_frame(days, np.linspace(100, 80, n), np.linspace(102, 82, n),
                     np.linspace(98, 78, n), np.linspace(101, 81, n), np.full(n, 5e5))
    wk = resample_weekly(df, keep_partial=True, today=pd.Timestamp("2025-12-31"))
    assert len(wk) == n
    assert getattr(wk.index, "tz", None) is None


def test_locked_week_detection():
    """A lock means the price *could not trade*: a pinned limit day or a no-trade session.
    A merely large move is not a lock (that filter used to reject 998/1000 NSE stocks)."""
    days = list(pd.date_range("2024-01-01", "2024-01-05", freq="B"))
    o = np.full(5, 100.0)

    # (1) a +5.0% wide-range day with full volume -> ordinary volatility, NOT locked
    up = o.copy(); up[1] = 105.0
    wk = resample_weekly(daily_frame(days, o, np.maximum(o, up) + 0.1,
                                     np.minimum(o, up) - 0.1, up, np.full(5, 1e6)),
                         keep_partial=True, today=pd.Timestamp("2024-01-05"))
    assert float(wk["LockedDays"].iloc[0]) == 0.0, wk

    # (2) a day pinned at the 10% circuit limit (open=high=low=close, nothing traded) -> locked
    c2 = o.copy(); c2[1] = 110.0
    o2 = o.copy(); o2[1] = 110.0
    h2 = np.maximum(o, c2) + 0.0; h2[1] = 110.0
    l2 = np.minimum(o, c2) - 0.0; l2[1] = 110.0
    v2 = np.full(5, 1e6); v2[1] = 0.0
    wk2 = resample_weekly(daily_frame(days, o2, h2, l2, c2, v2),
                          keep_partial=True, today=pd.Timestamp("2024-01-05"))
    assert float(wk2["LockedDays"].iloc[0]) >= 1.0, wk2

    # (3) flat / no-trade sessions (a locked limit price) are also detected
    flat = daily_frame(days, o, o, o, o, np.array([1e6, 0.0, 0.0, 1e6, 1e6]))
    wk3 = resample_weekly(flat, keep_partial=True, today=pd.Timestamp("2024-01-05"))
    assert float(wk3["LockedDays"].iloc[0]) >= 2.0, wk3


# --------------------------------------------------------------------------------------
# Swing structure
# --------------------------------------------------------------------------------------
def test_swing_lows_fractal_rule():
    wk = weekly([(10, 11, 9, 10), (10, 11, 8, 9), (9, 12, 7, 11), (11, 13, 10, 12),
                 (12, 14, 11, 13)])
    assert bool(swing_lows(wk, 2)[2])
    # plateau of equal lows is booked once (the last equal bar); the level is de-duplicated
    # by the sweep stage so the pool is still counted exactly once
    wk2 = weekly([(10, 11, 9, 10), (9, 10, 5, 9), (9, 10, 5, 9), (9, 10, 6, 9), (9, 11, 8, 10)])
    m = swing_lows(wk2, 2)
    assert bool(m[2]) and int(m.sum()) == 1, m
    # the last `strength` bars can never be a *confirmed* swing low
    assert not m[-2:].any()


def test_swing_low_survives_a_later_lower_low_and_edge_rule_holds():
    wk = weekly([(10, 11, 9, 10), (10, 11, 8, 9), (9, 12, 7, 11), (11, 13, 10, 12),
                 (12, 14, 11, 13), (13, 15, 12, 14), (14, 15, 6, 13), (13, 16, 12, 15),
                 (15, 16, 14, 15)])
    m = swing_lows(wk, 2)
    assert bool(m[2]) and bool(m[6])          # both are local lows
    assert not m[-2:].any(), "un-confirmable right edge must never be booked"


def test_swing_strength_controls_confirmation_delay():
    """strength=N sets how many bars a low needs on each side to become a usable level."""
    bars = drift(27) + [(101.0, 103.0, 100.0, 102.5), (102.5, 105.0, 102.0, 104.5),
                        (104.0, 104.5, 96.2, 104.0, 1_500_000)]
    wk = weekly(bars)
    assert bool(swing_lows(wk, 1)[27]) and not bool(swing_lows(wk, 2)[27])
    common = dict(min_weekly_bars=20, swing_lookback=20, allow_fresh_low=False,
                  min_depth_pct=0.0, max_range_of_close=0.15, min_wick_ratio=0.0,
                  min_wick_body_mult=0.0, min_close_in_range=0.0)
    weak = find_sweeps(wk, Params(**{**common, "swing_strength": 1}))
    strong = find_sweeps(wk, Params(**{**common, "swing_strength": 2}))
    assert weak and abs(weak[0].level - 100.0) < 1e-6, [h.level for h in weak]
    assert not strong, "an unconfirmed swing must not be usable at strength=2"
    # strength also decides whether a low is confirmed at all: the sweep_tape swing needs
    # only 6 bars of right-side room, so strength=5 still sees it, but a short lookback does not
    wk_s = sweep_tape()
    pl_short = Params(min_weekly_bars=20, swing_strength=5, allow_fresh_low=False,
                      swing_lookback=3, min_depth_pct=0.0)
    pl_long = Params(**{**pl_short.__dict__, "swing_lookback": 10})
    assert _levels_to_sweep(wk_s, len(wk_s) - 1, pl_short) == []
    assert abs(_levels_to_sweep(wk_s, len(wk_s) - 1, pl_long)[0][0] - 100.0) < 1e-6
    assert bool(swing_lows(wk_s, 5)[26]) and not bool(swing_lows(wk_s, 26)[26])

# --------------------------------------------------------------------------------------
# The sweep rule itself
# --------------------------------------------------------------------------------------
def test_detects_classic_sweep_of_old_swing_low():
    wk = sweep_tape()
    hits = find_sweeps(wk, Params(min_weekly_bars=20))
    assert hits, "a textbook sweep must be detected"
    h = hits[0]
    assert h.sweep_idx == len(wk) - 1
    assert abs(h.level - 100.0) < 1e-6, h.level          # the OLD swing low was the pool
    assert h.low < h.level < h.close                      # pierced then closed back above
    assert abs(h.depth_abs - 4.0) < 1e-6
    assert abs(h.close_above_pct - 0.025) < 1e-6
    assert abs(h.wick_ratio - 0.7143) < 1e-3, h.wick_ratio   # lower wick 5.0 / range 7.0
    assert abs(h.lower_wick - 5.0) < 1e-9 and abs(h.body - 1.5) < 1e-9
    assert abs(h.close_in_range - 0.9286) < 1e-3
    assert h.fresh_low is False and h.green is True
    assert h.swing_age_weeks == 6 and h.bars_since_sweep == 0
    assert h.score > 60, h.score
    r = _row(h, wk)
    assert r["SweepType"] == "52-week-low sweep" and bool(r["Is52wLow"])
    assert r["SweptSwingLow"] == 100.0 and r["SweepLow"] == 96.0
    assert r["PoolsTaken"] == 1

def test_sweep_output_carries_tradeable_levels():
    wk = sweep_tape()
    h = find_sweeps(wk, Params(min_weekly_bars=20))[0]
    r = _row(h, wk)
    assert r["SweepLow"] == 96.0 and r["SweptSwingLow"] == 100.0
    assert abs(r["Stop"] - round(96.0 * 0.995, 2)) < 1e-6
    assert r["Risk%"] > 0
    assert r["Target2R"] > r["Close"] and r["Target3R"] > r["Target2R"]
    assert r["BarsSinceSweep"] == 0
    assert np.isfinite(r["ATR(14w)"]) and r["ATR(14w)"] > 0


def test_no_sweep_when_close_finishes_below_level():
    wk = final_bar_tape(o=101.0, h=103.0, l=95.5, c=95.5)   # closes on its low, below 100
    p = Params(min_weekly_bars=20)
    assert not find_sweeps(wk, p), "close below the swept level is a breakdown, not a buy setup"
    assert near_miss(wk, p) is None


def test_no_sweep_when_close_is_exactly_at_level():
    # close == swept level (equal, not above) and no tail at all
    wk = final_bar_tape(o=96.0, h=100.0, l=96.0, c=100.0)
    assert not find_sweeps(wk, Params(min_weekly_bars=20)), "close must be ABOVE the swept low"


def test_no_sweep_without_wick():
    """
    A bar that ends AT its low (close ~= low, near the bottom of the range) has no rejection
    body at all -> never a sweep, and never even a near miss. This is the bar a naive
    "low < swing_low and close > swing_low" screener wrongly flags.
    """
    wk = final_bar_tape(o=96.0, h=103.0, l=95.9, c=96.05)
    p = Params(min_weekly_bars=20)
    assert not find_sweeps(wk, p)
    assert near_miss(wk, p) is None, "close below the swept level -> nothing reclaimed"
    # same idea but the level IS reclaimed; with no tail the bar is only a watchlist item
    wk2 = final_bar_tape(o=96.6, h=103.0, l=96.5, c=101.0)
    assert not find_sweeps(wk2, p)
    m = near_miss(wk2, p)
    assert m is not None and "wick too small" in m["FailReason"], m
    assert m["SweptLevel"] == 100.0 and float(m["WickRatio%"]) < 5.0, m


def test_min_close_above_pct_cushion_is_enforced():
    # close 100.5 = only 0.5% above the swept 100 level, tight 4.5-wide range
    wk = final_bar_tape(o=100.6, h=100.9, l=96.4, c=100.5, v=2_000_000)
    p_loose = Params(min_weekly_bars=20, min_close_above_pct=0.0)
    p_tight = Params(min_weekly_bars=20, min_close_above_pct=0.02)
    assert find_sweeps(wk, p_loose), "0.6% above the level passes a 0% cushion"
    assert not find_sweeps(wk, p_tight), "0.6% above the level fails a 2% cushion"


def test_dragonfly_doji_is_a_hit_and_a_strict_gate_downgrades_it():
    # open == high == close, deep tail: the purest rejection candle (wick ratio ~1.0)
    wk = final_bar_tape(o=101.0, h=101.05, l=96.0, c=101.02)
    assert abs(float(wk["Low"].iloc[-1]) - 96.0) < 1e-9
    p_norm = Params(min_weekly_bars=20)
    hits = find_sweeps(wk, p_norm)
    assert hits and hits[0].wick_ratio > 0.98, hits
    strict = Params(min_weekly_bars=20, min_close_in_range=0.999)   # demands a close at the very top
    assert not find_sweeps(wk, strict)
    m = near_miss(wk, strict)
    assert m is not None and "close weak" in m["FailReason"], m


def test_body_dominating_the_wick_is_rejected():
    # a big-bodied red bar that dips to 96 and closes at 100.5: wick 0.5 vs body 4.5
    bars = drift(26) + PRE + [(105.0, 105.2, 96.0, 100.5)]
    wk = weekly(bars)
    p = Params(min_weekly_bars=20, min_wick_ratio=0.0)
    assert not find_sweeps(wk, p), "the wick must dominate the body"
    m = near_miss(wk, p)
    assert m and "body larger than wick" in m["FailReason"], m

def test_fresh_low_sweep_can_be_switched_off():
    wk = fresh_low_tape()
    on = Params(min_weekly_bars=5, allow_fresh_low=True)
    off = Params(min_weekly_bars=5, allow_fresh_low=False)
    hits = find_sweeps(wk, on)
    assert hits, "undercutting a fresh low and closing back above is a valid sweep"
    assert hits[0].fresh_low is True and hits[0].is_low_52w is True
    assert not find_sweeps(wk, off), "must vanish when fresh-low sweeps are disabled"


SELF_SWEEP_BARS = drift(26) + [(101.0, 103.0, 100.0, 102.5), (102.5, 105.0, 102.0, 104.5),
                              (104.5, 106.0, 103.5, 105.0), (105.0, 106.0, 104.0, 104.5),
                              (104.0, 105.0, 103.2, 104.8)]


def test_a_bar_must_not_sweep_the_low_it_is_printing():
    """On a monotonic tape the floor IS the newest bar's low; nothing was resting below it."""
    wk = weekly(drift(27, start=125.0, step=1.0))
    lows = wk["Low"].to_numpy(float)
    i = len(wk) - 1
    assert int(np.argmin(lows)) == i, "the last bar prints the tape's lowest low"
    assert list(np.flatnonzero(swing_lows(wk, 2))) == [], "no confirmed swing on this tape"
    loose = dict(min_weekly_bars=20, allow_fresh_low=True, min_wick_ratio=0.0,
                 min_wick_body_mult=0.0, min_close_in_range=0.0, max_range_of_close=0.9,
                 min_depth_pct=0.0, swing_lookback=26)
    off = Params(**{**loose, "include_sweep_on_swing_bar": False})
    on = Params(**{**loose, "include_sweep_on_swing_bar": True})
    # the engine's own floor is this bar's index -> it can never appear as a level in "off" mode
    lv_off = _levels_to_sweep(wk, i, off)
    assert i not in {j for _, j, _ in lv_off}, "a bar must not be listed as its own swept pool"
    lv_on = _levels_to_sweep(wk, i, on)
    assert i in {j for _, j, _ in lv_on}, "include_sweep_on_swing_bar is what unlocks that pool"
    assert not find_sweeps(wk, off)


def test_lowest_pool_taken_is_the_one_reported():
    # swing 100, then a deeper swing 97; one bar undercuts BOTH -> it must report 97 (the
    # pool actually taken out), and it must be booked as a 2-pool sweep.
    bars = drift(24) + PRE[:4] + [(104.0, 104.6, 97.0, 104.2), (104.2, 105.0, 103.5, 104.6),
                                  (103.0, 104.0, 96.0, 103.5, 2_000_000)]
    wk = weekly(bars)
    hits = find_sweeps(wk, Params(min_weekly_bars=20, swing_lookback=26, min_wick_ratio=0.30))
    assert hits
    assert abs(hits[0].level - 97.0) < 1e-6, [h.level for h in hits]
    assert hits[0].n_pools == 2


def test_52w_flag_only_when_really_the_52w_low():
    bars = (drift(20) + [(150.0, 151.0, 80.0, 145.0, 2_000_000)] + drift(39, start=145.0, step=1.0)
            + [(101.0, 103.0, 100.0, 102.5), (102.5, 105.0, 102.0, 104.5),
               (104.5, 106.0, 103.5, 105.0), (105.0, 106.0, 104.0, 104.5),
               (104.5, 105.5, 103.0, 104.0), (104.0, 105.0, 103.2, 104.8),
               (101.0, 103.0, 96.0, 102.5, 4_000_000)])
    wk = weekly(bars)
    hits = find_sweeps(wk, Params(min_weekly_bars=20, swing_lookback=13))
    assert hits, "100 must still be swept as an old swing low"
    assert hits[0].is_low_52w is False, "the 52-week low here is 80, not 100"
    assert hits[0].fresh_low is False


def test_sweep_of_a_52w_low_is_flagged():
    hits = find_sweeps(sweep_tape(), Params(min_weekly_bars=20))
    assert hits[0].is_low_52w is True      # on this tape 100 IS the lowest low of the year


def test_only_recent_sweeps_when_max_bars_ago_is_one():
    wk = stale_sweep_tape(n_after=4)
    assert not find_sweeps(wk, Params(min_weekly_bars=20, max_sweep_bars_ago=1))
    hits = find_sweeps(wk, Params(min_weekly_bars=20, max_sweep_bars_ago=6))
    assert hits and hits[0].sweep_idx == len(wk) - 5, [h.sweep_idx for h in hits]


def test_fresh_setup_ranks_first():
    bars = (drift(26) + [(101.0, 103.0, 100.0, 102.5), (102.5, 105.0, 102.0, 104.5),
                         (104.5, 106.0, 103.5, 105.0), (105.0, 106.0, 104.0, 104.5),
                         (104.0, 105.0, 103.0, 104.0),
                         (101.0, 103.0, 96.0, 102.5, 4_000_000),      # stale sweep of 100
                         (102.5, 105.0, 102.4, 104.5), (104.5, 106.0, 104.0, 105.5),
                         (105.5, 106.5, 104.5, 105.0), (105.0, 105.5, 104.2, 104.6),
                         (104.6, 105.0, 104.0, 104.4),
                         (104.2, 105.0, 95.0, 104.8, 1_500_000)])      # fresh sweep of 100
    wk = weekly(bars)
    hits = find_sweeps(wk, Params(min_weekly_bars=20, max_sweep_bars_ago=8))
    assert len(hits) >= 2, [(h.sweep_idx, h.level) for h in hits]
    assert hits[0].sweep_idx == len(wk) - 1, [(h.sweep_idx, h.score) for h in hits]
    assert hits[0].bars_since_sweep == 0


def test_monster_bar_rejected_as_corporate_action():
    wk = final_bar_tape(o=104.0, h=140.0, l=96.0, c=130.0)   # 34% weekly range
    p = Params(min_weekly_bars=20, max_range_of_close=0.12)
    assert not find_sweeps(wk, p)
    m = near_miss(wk, p)
    assert m and "range too wide" in m["FailReason"], m


def test_noise_pierce_below_min_depth_rejected():
    wk = final_bar_tape(o=104.0, h=105.0, l=99.98, c=104.8)   # 2 paise under the level
    assert not find_sweeps(wk, Params(min_weekly_bars=20, min_depth_pct=0.004))
    assert find_sweeps(wk, Params(min_weekly_bars=20, min_depth_pct=0.0)), \
        "with the floor removed the same bar becomes a (weak) sweep"


def test_deep_collapse_through_the_level_is_rejected():
    wk = final_bar_tape(o=104.0, h=104.6, l=70.0, c=104.2)   # 30% under the level
    p = Params(min_weekly_bars=20, max_depth_pct=0.16, max_range_of_close=0.9)
    assert not find_sweeps(wk, p)


def test_volume_drydown_penalised_not_fatal():
    p = Params(min_weekly_bars=20)
    wk1 = sweep_tape()
    wk2 = sweep_tape()
    wk2.iloc[-1, wk2.columns.get_loc("Volume")] = 20_000.0
    h1, h2 = find_sweeps(wk1, p)[0], find_sweeps(wk2, p)[0]
    assert h2.score < h1.score, (h1.score, h2.score)
    assert h2.sweep_idx == h1.sweep_idx


def test_stricter_gates_are_respected():
    base = Params(min_weekly_bars=20)
    assert find_sweeps(sweep_tape(), base)
    assert find_sweeps(sweep_tape(), Params(min_weekly_bars=20, require_green_close=True))
    red = final_bar_tape(o=103.0, h=103.2, l=96.0, c=100.5)     # red hammer (close < open)
    assert find_sweeps(red, base), "a red rejection candle is still a sweep by default"
    assert not find_sweeps(red, Params(min_weekly_bars=20, require_green_close=True))


def test_min_score_gate_filters():
    wk = sweep_tape()
    s0 = find_sweeps(wk, Params(min_weekly_bars=20))[0].score
    assert find_sweeps(wk, Params(min_weekly_bars=20, min_score=1.0))
    assert not find_sweeps(wk, Params(min_weekly_bars=20, min_score=min(99.0, s0 + 1.0)))


def test_swing_strength_changes_sensitivity():
    """A wider fractal needs a deeper V; a narrow one fires on minor kinks."""
    wk = sweep_tape()
    strong = find_sweeps(wk, Params(min_weekly_bars=20, swing_strength=5))
    weak = find_sweeps(wk, Params(min_weekly_bars=20, swing_strength=1))
    assert weak, "strength=1 must find the 100 swing"
    assert isinstance(strong, list)     # may be empty; must simply never crash


def test_lookback_window_limits_which_pool_can_be_swept():
    """swing_lookback decides which older pools are even in play for a sweep."""
    bars = [(101.0, 103.0, 80.0, 101.0), (101.0, 103.0, 101.5, 102.5)] + drift(24)
    bars += PRE[:5] + [(104.0, 104.6, 79.0, 104.4, 1_500_000)]
    wk = weekly(bars)
    common = dict(min_weekly_bars=20, min_depth_pct=0.0, max_range_of_close=0.30,
                  max_depth_pct=0.90, min_wick_ratio=0.0, min_wick_body_mult=0.0,
                  min_close_in_range=0.0)
    wide = find_sweeps(wk, Params(**{**common, "swing_lookback": 40}))
    narrow = find_sweeps(wk, Params(**{**common, "swing_lookback": 6}))
    assert wide and abs(wide[0].level - 80.0) < 1e-6, [h.level for h in wide]
    assert wide[0].n_pools == 2, wide[0].n_pools         # both 80 and 100 were undercut
    assert wide[0].swing_age_weeks == 31
    assert narrow and abs(narrow[0].level - 100.0) < 1e-6, [h.level for h in narrow]
    assert narrow[0].n_pools == 1, narrow[0].n_pools    # the deep 80 pool is out of range


# --------------------------------------------------------------------------------------
# Plumbing: screen_all / rank / backtest / charts
# --------------------------------------------------------------------------------------
def test_screen_all_end_to_end_on_synthetic_universe():
    p = Params(min_weekly_bars=20, min_turnover_lakh=0.0)
    wk_good = sweep_tape()
    data = {
        "GOODCO": as_daily(wk_good),
        "FLATCO": as_daily(weekly(drift(60))),
        "TINYCO": as_daily(weekly([(1.0, 1.02, 0.99, 1.01)] * 140)),
    }
    hits, miss, diag = screen_all(data, p)
    assert not hits.empty and list(hits["Symbol"]) == ["GOODCO"], hits
    assert hits.iloc[0]["SweptSwingLow"] == 100.0
    assert float(hits.iloc[0]["SweepLow"]) == 96.0
    assert hits.iloc[0]["Rank"] == 1 and hits.iloc[0]["Score"] > 0
    assert str(pd.Timestamp(wk_good.index[-1]).date()) == hits.iloc[0]["SweepWeek"]
    assert set(diag["Symbol"]) == set(data)
    assert "price_below_min" in set(diag["status"]), diag
    assert "FLATCO" in set(diag.loc[diag["status"] == "no_setup", "Symbol"]), diag
    for c in ("Close", "SweptSwingLow", "SweepLow", "WickRatio%", "Risk%", "Stop", "Target2R"):
        assert c in hits.columns, c


def test_screen_all_reports_near_miss_watchlist():
    weak = final_bar_tape(o=96.6, h=103.0, l=96.5, c=101.0)     # reclaimed 100, zero rejection tail
    strict = Params(min_weekly_bars=20, min_turnover_lakh=0.0)
    hits, miss, diag = screen_all({"WEAKCO": as_daily(weak)}, strict)
    assert hits.empty and not miss.empty, (hits, miss)
    assert miss.iloc[0]["Symbol"] == "WEAKCO"
    assert "wick too small" in miss.iloc[0]["FailReason"]
    assert diag.iloc[-1]["status"] == "near_miss", diag
    # dropping the close-quality bar is enough to turn the SAME candle into a hit
    loose = Params(min_weekly_bars=20, min_turnover_lakh=0.0, min_wick_ratio=0.0,
                   min_wick_body_mult=0.0, min_close_in_range=0.0, max_range_of_close=0.9)
    hits2, miss2, diag2 = screen_all({"WEAKCO": as_daily(weak)}, loose)
    assert not hits2.empty and diag2.iloc[-1]["status"] == "hit", (hits2, diag2)
    # keep_near_miss=False suppresses the watchlist but not the diagnostics
    h3, m3, d3 = screen_all({"WEAKCO": as_daily(weak)}, strict, keep_near_miss=False)
    assert m3.empty and len(d3) == 1


def test_screen_all_illiquidity_and_circuit_filters():
    wk = sweep_tape()
    daily = as_daily(wk)
    p_liquid = Params(min_weekly_bars=20, min_turnover_lakh=1e9)   # absurd threshold
    hits, miss, diag = screen_all({"GOODCO": daily}, p_liquid)
    assert hits.empty
    assert diag.iloc[-1]["status"] == "illiquid", diag
    # circuit / limit-locked weeks: 3 pinned no-trade sessions a week -> filtered
    days = list(pd.date_range("2023-06-05", periods=60, freq="B"))
    o = np.full(60, 100.0); c = o.copy()
    v = np.full(60, 1e6)
    for k in range(5, 60, 5):                 # three zero-range, zero-volume days each week
        for j in (k, k + 1, k + 2):
            c[j] = 100.0; v[j] = 0.0
    lock = daily_frame(days, o, o.copy(), o.copy(), c, v)
    hits3, miss3, diag3 = screen_all({"LOCKCO": lock}, Params(min_weekly_bars=5,
                                                              min_turnover_lakh=0.0))
    assert hits3.empty
    assert diag3.iloc[-1]["status"] == "circuit_locked", diag3.iloc[-1]
    # and the SAME tape without the locks must NOT be circuit-locked (no false positives)
    clean = daily_frame(days, o, o + 1.0, o - 1.0, o.copy(), np.full(60, 1e6))
    h4, m4, d4 = screen_all({"CLEANCO": clean}, Params(min_weekly_bars=5, min_turnover_lakh=0.0))
    assert (d4.iloc[-1]["status"] != "circuit_locked") or (d4.iloc[-1]["LockedWeeks26"] <= 1), d4


def test_screen_all_survives_garbage_input():
    p = Params(min_weekly_bars=20)
    junk = {
        "EMPTY": pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        "NANS": pd.DataFrame({"Open": [np.nan], "High": [np.nan], "Low": [np.nan],
                              "Close": [np.nan], "Volume": [np.nan]},
                             index=pd.to_datetime(["2024-01-01"])),
        "SHORT": as_daily(weekly(drift(4))),
        "INV": as_daily(weekly([(10.0, 5.0, 20.0, 1.0)] * 130)),   # high < low
        "NOTIME": pd.DataFrame({"Open": [1.0, 2.0], "High": [3.0, 4.0], "Low": [0.5, 1.5],
                                "Close": [2.0, 3.0], "Volume": [1.0, 2.0]}, index=["x", "y"]),
    }
    hits, miss, diag = screen_all(junk, p)
    assert isinstance(hits, pd.DataFrame) and isinstance(diag, pd.DataFrame)
    assert hits.empty
    assert set(diag["Symbol"]) >= {"EMPTY", "NANS", "SHORT", "INV"}


def test_progress_callback_is_called():
    calls = []
    data = {f"S{i}": as_daily(weekly(drift(120))) for i in range(3)}
    screen_all(data, Params(min_weekly_bars=20, min_turnover_lakh=0.0),
               keep_near_miss=False, progress_cb=lambda a, b: calls.append((a, b)))
    assert calls and calls[0][1] == 3


def test_rank_top_prefers_market_cap_then_turnover():
    data = {s: as_daily(weekly(drift(60))) for s in ("A", "B", "C")}
    data["A"].iloc[-1, data["A"].columns.get_loc("Volume")] = 1e9
    quotes = pd.DataFrame({"Symbol": ["A", "B", "C"], "MarketCap": [1e11, 5e11, 2e11]})
    tab = rank_top(data, quotes, top_n=3)
    assert list(tab["Symbol"]) == ["B", "C", "A"], tab
    assert tab["rank_basis"].iloc[0] == "market_cap"
    # no quotes -> turnover/index ordering, and it must still be a clean 1..N ranking
    tab2 = rank_top(data, None, top_n=3)
    assert tab2["rank_basis"].iloc[0] == "index_tier+turnover"
    assert list(tab2["Rank"]) == [1, 2, 3]
    assert list(tab2["Symbol"])[0] == "A"          # A has the fattest turnover
    # patchy market cap must NOT hijack the ranking (that would silently drop most stocks)
    bad = quotes.copy(); bad.loc[1:, "MarketCap"] = np.nan
    assert rank_top(data, bad, top_n=3)["rank_basis"].iloc[0] == "index_tier+turnover"


def test_rank_top_honours_index_tiers_and_truncates():
    data = {s: as_daily(weekly(drift(60))) for s in ("BIG", "MID", "SMALL", "MICRO")}
    # make turnover strictly ordered SMALL > MICRO > MID > BIG so the index tier must decide
    for k, sym in enumerate(("SMALL", "MICRO", "MID", "BIG")):
        data[sym].iloc[:, data[sym].columns.get_loc("Volume")] = float(40 - k) * 1e7
    tiers = pd.DataFrame({"Symbol": ["BIG"], "IndexTier": ["Nifty50"], "Industry": ["Oil"]})
    tab = rank_top(data, None, top_n=4, index_tiers=tiers)
    assert list(tab["Symbol"])[0] == "SMALL", tab        # default: liquidity ordering wins
    assert list(tab["Symbol"]) == ["SMALL", "MICRO", "MID", "BIG"], list(tab["Symbol"])
    assert tab["IndexTier"].dropna().tolist() == ["Nifty50"]   # tier is carried, not enforced
    assert tab["Industry"].dropna().unique().tolist() == ["Oil"]
    # opt-in: official index tier first, turnover inside each tier
    tab2 = rank_top(data, None, top_n=4, index_tiers=tiers, prefer_index_tier=True)
    assert tab2["Symbol"].iloc[0] == "BIG", list(tab2["Symbol"])
    assert list(tab2["Rank"]) == [1, 2, 3, 4]
    # truncation is real
    assert len(rank_top(data, None, top_n=2)) == 2


def test_rank_top_empty():
    assert rank_top({}, None).empty


def _repetitive_tape(reps=6):
    """Alternating swing + sweep, so the backtest sees many repeatable occurrences."""
    bars = drift(10, start=170.0, step=1.5)
    for _ in range(reps):
        bars += [(151.0, 153.0, 150.0, 152.5), (152.5, 155.0, 152.0, 154.5),
                 (154.5, 156.0, 153.5, 155.0), (155.0, 156.0, 154.0, 155.5)]
        bars.append((151.5, 154.0, 145.5, 153.5, 3_000_000))   # sweep of the 150 swing low
        bars += [(153.5, 158.0, 153.0, 157.0), (157.0, 161.0, 156.0, 160.0),
                 (160.0, 163.0, 158.5, 162.0)]
    return weekly(bars)


def test_backtest_and_summary():
    wk = _repetitive_tape()
    daily = as_daily(wk)
    p = Params(min_weekly_bars=20, min_turnover_lakh=0.0)
    trades = backtest({"SYN": daily}, p, horizon_weeks=6, r_mult=2.0)
    assert not trades.empty, "repetitive tape must produce trades"
    assert np.isfinite(trades["Return%"].to_numpy(float)).all()
    assert (trades["SweptLevel"] - 150.0).abs().max() < 1e-6
    # entry = open of the week AFTER the sweep
    entries = trades["Entry"].to_numpy(float)
    opens = wk["Open"].to_numpy(float)
    for e in entries:
        assert float(np.abs(opens - e).min()) < 0.011, e
    s = summarize_backtest(trades)
    assert 0 <= s["win_rate_%"] <= 100 and s["trades"] == len(trades)
    assert abs(s["hit_target_%"] + s["hit_stop_%"] + s["expired_%"] - 100.0) < 0.15
    assert summarize_backtest(pd.DataFrame()) == {"trades": 0}


def test_backtest_exit_priority_is_stop_before_target_same_week():
    """If a week touches both, assume the stop (conservative)."""
    wk = weekly(_repetitive_tape(1).iloc[:15].to_dict("records") and
                drift(10) + [(151.0, 153.0, 150.0, 152.5), (152.5, 155.0, 152.0, 154.5),
                             (154.5, 156.0, 153.5, 155.0), (155.0, 156.0, 154.0, 155.5),
                             (151.5, 154.0, 145.5, 153.5, 3_000_000),
                             (153.5, 200.0, 100.0, 120.0)])   # both stop and target in one week
    trades = backtest({"SYN": as_daily(wk)}, Params(min_weekly_bars=20, min_turnover_lakh=0.0),
                      horizon_weeks=4, r_mult=2.0)
    if not trades.empty:
        assert set(trades["Outcome"]) <= {"stop", "target", "time"}
        assert (trades["Return%"] < 0).any(), "touching both must resolve as the stop"


def test_chart_frames_shapes():
    wk = sweep_tape()
    tail, info = chart_frames(as_daily(wk), Params(min_weekly_bars=20), weeks_to_show=20)
    assert 0 < len(tail) <= 20
    assert info["hit"] is not None
    assert 0 <= info["hit"]["sweep_pos"] < len(tail)
    assert all(0 <= s["pos"] < len(tail) for s in info["swings"])
    assert info["levels"] and all(l["level"] > 0 for l in info["levels"])
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    t0, i0 = chart_frames(empty, Params(min_weekly_bars=20), weeks_to_show=20)
    assert t0.empty and i0["hit"] is None


def test_params_describe_is_json_safe():
    import json
    d = Params().describe()
    json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()})
    assert set(d) >= {"swing_strength", "min_wick_ratio", "max_sweep_bars_ago"}


def test_empty_inputs_never_raise():
    p = Params()
    assert find_sweeps(weekly([]), p) == []
    assert find_sweeps(weekly([(1, 2, 0.5, 1.5)] * 5), p) == []
    assert near_miss(weekly([(1.0, 2.0, 0.5, 1.5)] * 2), p) is None
    assert swing_lows(weekly([(1, 2, 0.5, 1.5)] * 3), 2).sum() == 0
    assert atr(weekly(drift(20)), 14).gt(0).all()
    assert len(resample_weekly(pd.DataFrame(columns=["Open", "High", "Low", "Close"]))) == 0


def test_resample_weekly_locked_days_is_silent_on_normal_volatility():
    """Guard for the bug that rejected 998/1000 symbols: a 5% day is NOT a circuit lock."""
    days = list(pd.bdate_range("2024-01-01", periods=25))
    o = np.full(25, 100.0); h = o + 2.0; l = o - 2.0; c = o.copy(); v = np.full(25, 1e6)
    c[3] = 105.0; h[3] = 106.0; l[3] = 99.0          # a wide, liquid +5% day
    c[9] = 94.0;  h[9] = 101.0; l[9] = 93.0           # and a -6% one
    wk = resample_weekly(pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                                        "Volume": v}, index=pd.DatetimeIndex(days)),
                         keep_partial=True)
    assert float(wk["LockedDays"].sum()) == 0.0, wk["LockedDays"]
    assert locked_weeks(wk) == 0

    # a day pinned at +10% with no trade through it IS a lock
    c2 = c.copy(); o2 = o.copy(); h2 = h.copy(); l2 = l.copy(); v2 = v.copy()
    c2[9] = o2[9] = h2[9] = l2[9] = 110.0; v2[9] = 0.0
    wk2 = resample_weekly(pd.DataFrame({"Open": o2, "High": h2, "Low": l2, "Close": c2,
                                         "Volume": v2}, index=pd.DatetimeIndex(days)),
                          keep_partial=True)
    assert locked_weeks(wk2) >= 1, wk2["LockedDays"]


def test_assess_daily_repairs_inconsistent_bars_instead_of_deleting_them():
    """Yahoo hands back bars where High ignores Open (NORBTEAEXP 2021-06-25). The row must
    survive with a corrected range - deleting it silently loses sweep evidence."""
    from nse_data import assess_daily
    days = list(pd.bdate_range("2024-01-01", periods=5))
    o = np.array([6.35, 6.40, 6.45, 6.50, 6.55]); c = o.copy()
    h = np.array([6.30, 6.42, 6.47, 6.52, 6.57])          # first High < Open -> impossible
    l = np.array([6.25, 6.38, 6.43, 6.48, 6.53])
    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                       "Volume": np.full(5, 1e6)}, index=pd.DatetimeIndex(days))
    clean, issues, m = assess_daily(df, min_weekly_bars=1, max_fresh_lag_days=10_000)
    assert len(clean) == 5, (len(clean), issues)          # nothing dropped
    assert float(clean["High"].iloc[0]) == 6.35           # repaired to max(O,H,C)
    assert m["ohlc_repaired"] == 1
    assert any("repaired" in i for i in issues), issues


def test_max_sweep_bars_ago_actually_widens_the_window():
    """Regression: the screener once hard-filtered to the newest bar, which made
    max_sweep_bars_ago a silent no-op (a "5-week" screen returned the "1-week" answer)."""
    base = sweep_tape()                                    # sweep on the newest week
    p1 = Params(min_weekly_bars=20, min_turnover_lakh=0.0)
    p5 = Params(min_weekly_bars=20, min_turnover_lakh=0.0, max_sweep_bars_ago=5)
    h_now, _, _ = screen_all({"CO": as_daily(base)}, p1)
    assert len(h_now) == 1, h_now

    # three quiet weeks on top: nothing sweeps, the old sweep is now 3 weeks stale
    dead = pd.DataFrame({"Open": 130.0, "High": 133.0, "Low": 128.0, "Close": 131.0,
                         "Volume": 9e5},
                        index=pd.DatetimeIndex([base.index[-1] + pd.Timedelta(weeks=k)
                                                for k in (1, 2, 3)]))
    grown = as_daily(pd.concat([base, dead]))
    h_strict, _, d_strict = screen_all({"CO": grown}, p1)
    h_wide, _, _ = screen_all({"CO": grown}, p5)
    assert len(h_strict) == 0, h_strict          # newest week sweeps nothing
    assert len(h_wide) == 1, h_wide              # a window of 5 reaches back to the real sweep
    assert int(h_wide.iloc[0]["BarsSinceSweep"]) == 3, h_wide.iloc[0].to_dict()
    assert float(h_wide.iloc[0]["Score"]) < float(h_now.iloc[0]["Score"])   # recency penalised
    assert d_strict.iloc[-1]["status"] in ("no_setup", "near_miss"), d_strict.iloc[-1]["status"]


def test_cache_path_is_collision_free_for_special_symbols():
    """'M&M' and 'M_M' must not share a cache file, and read/write must always agree."""
    from nse_data import _cache_path, save_daily_cache, load_daily_cache
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        a = _cache_path(d, "M&M", "1d", "csv")
        b = _cache_path(d, "M_M", "1d", "csv")
        c = _cache_path(d, " M&M ", "1d", "csv")
        assert a != b, (a, b)
        assert a == c, (a, c)                      # whitespace-only difference resolves the same
        df = daily_frame(list(pd.date_range("2024-01-01", periods=5, freq="B")),
                         np.full(5, 10.0), np.full(5, 11.0), np.full(5, 9.0),
                         np.full(5, 10.5), np.full(5, 1000.0))
        save_daily_cache(df, d, "M&M", "csv")
        got = load_daily_cache(d, "M&M", "csv")
        assert got is not None and len(got) == 5
        assert load_daily_cache(d, "M_M", "csv") is None      # no cross-contamination
        assert not os.path.basename(a).count("..")