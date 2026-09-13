"""
config.py — parameters for the NSE Weekly Liquidity-Sweep Screener.

Copied VERBATIM from cell "2 · Parameters" of
NSE_Weekly_Liquidity_Sweep_Screener.ipynb so the automated run reproduces the
notebook result exactly. Edit here and the Saturday job picks it up.
"""

CONFIG = {
    # ---------- universe & data ----------
    "universe_size": 1000,        # screen the top-N by liquidity after ranking all of NSE
    "history_years": 6,           # 6y of daily bars -> ~310 weekly candles
    "max_workers": 3,             # parallel fetchers. 3 is the sweet spot vs. 429s
    "request_sleep": 0.12,        # GLOBAL min gap between any two HTTP calls (seconds)
    "throttle_pause": 20,         # park time when 429s cluster
    "final_pass": True,           # slow retry for throttled symbols on another transport
    "transport": "auto",          # auto | curl_cffi | requests | yfinance
    "use_quotes": False,          # True = also try Yahoo quotes for market cap (often 429)
    "prefer_index_tier": False,   # True = Nifty50/100/200... first, then liquidity
    "cache_dir": "cache_nse",
    "mount_drive": False,         # True = keep the cache + outputs on Google Drive

    # ---------- the sweep rule ----------
    "min_weekly_bars": 104,       # >= 2 years of weekly candles
    "min_price": 5.0,             # rupees; filters penny/illiquid noise
    "min_turnover_lakh": 75.0,    # median daily turnover, in lakh (75 L = 0.75 Cr)
    "swing_strength": 2,          # fractal: N bars lower on BOTH sides -> a confirmed swing
    "swing_lookback": 26,         # weeks back to hunt for swing lows
    "min_swing_age": 2,           # a swing needs this many closed weeks to be "confirmed"
    "allow_fresh_low": True,      # NEW lows count (window low / 52w low) - your requirement
    "require_old_swing": False,   # True = only pre-existing swing lows, no fresh lows
    "max_sweep_bars_ago": 1,      # 1 = sweep on the most recent CLOSED week. 3 = last 3 weeks
    "include_sweep_on_swing_bar": True,   # the sweep candle may BE the new swing low
    "min_close_above_pct": 0.002, # close must clear the swept level by 0.2%
    "min_wick_ratio": 0.34,       # lower wick >= 34% of the candle range
    "min_wick_body_mult": 1.15,   # lower wick >= 1.15x the body  (rejection, not a doji)
    "min_close_in_range": 0.50,   # close in the upper 50% of the range
    "min_depth_pct": 0.002,       # it must actually pierce the level by 0.2%...
    "max_depth_pct": 0.16,        # ...but not collapse 16% below it (that's a breakdown)
    "max_depth_atr_mult": 3.0,    # ...and not more than 3 weekly ATRs deep
    "max_range_of_close": 0.12,   # candle range <= 12% of close (no circuit-day lottery)
    "require_green_close": False, # True = only green sweep candles
    "exclude_locked_weeks": True, # drop weeks that gapped limit-up/down (untradeable)
    "max_locked_weeks_26": 1,
    "min_score": 0.0,             # 0 = show everything; 55 = only A-grade setups
}

# ---------- report / email knobs (automation only, not part of the screen) ----------
REPORT = {
    "gallery_symbols": 12,   # charts embedded in the HTML report (notebook uses 12)
    "gallery_weeks": 70,     # weeks of history per chart (notebook §6 uses 70)
    "table_rows": 40,        # rows in the emailed setups table
    "near_miss_rows": 15,    # near-misses shown in the report
}
