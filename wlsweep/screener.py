"""
screener.py — orchestration, presentation and exports for the NSE weekly sweep screener.

Everything network/CSV/plot-specific lives here so the two "engine" modules stay pure:
    nse_data.py     -> universe + bars (fetching is the hard part, so it is isolated here)
    sweep_engine.py -> weekly resampling, swing/sweep detection, scoring, backtest
    screener.py     -> the pipeline, tables, charts, Excel/CSV export, alerts  (this file)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nse_data as ND
import sweep_engine as SE
from nse_data import FetchConfig, YahooClient
from sweep_engine import Params

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def today_ts() -> pd.Timestamp:
    return pd.Timestamp(now_ist().date())


# ======================================================================================
# 1. Setup helpers (Colab-friendly)
# ======================================================================================
def ensure_packages(packages: Sequence[str] = ("pandas", "numpy", "requests", "tqdm",
                                               "openpyxl", "plotly")) -> List[str]:
    """
    Return the list of packages that are missing (the notebook pip-installs those and then
    asks for one runtime restart, which is the #1 cause of 'my colab notebook broke').
    """
    import importlib
    missing = []
    alias = {"plotly": "plotly", "openpyxl": "openpyxl", "yfinance": "yfinance"}
    for pkg in packages:
        mod = alias.get(pkg, pkg)
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    return missing


def mount_drive_if_requested(enabled: bool, mount_path: str = "/content/drive"):
    if not enabled:
        return None
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mount_path)
        return mount_path
    except Exception as exc:
        print(f"Drive mount skipped ({type(exc).__name__}: {exc}) - using local disk.")
        return None


def snapshot_workspace(workdir: str, dest_dir: str) -> Optional[str]:
    """Zip the notebook workspace (code + outputs, no raw bars) so a Colab disconnect is not fatal."""
    if not os.path.isdir(workdir):
        return None
    os.makedirs(os.path.dirname(dest_dir) or ".", exist_ok=True)
    keep = {".py", ".csv", ".md", ".txt", ".html", ".xlsx", ".json"}
    tmp = dest_dir + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(workdir):
            dirs[:] = [d for d in dirs if d not in {"cache_nse", "__pycache__", ".ipynb_checkpoints"}]
            for f in files:
                if os.path.splitext(f)[1].lower() in keep:
                    fp = os.path.join(root, f)
                    zf.write(fp, os.path.relpath(fp, workdir))
    shutil.move(tmp, dest_dir)
    return dest_dir


# ======================================================================================
# 2. Universe
# ======================================================================================
def build_universe(cfg: FetchConfig, client: YahooClient, log: Callable[[str], None] = print
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """NSE equity universe + index tier/industry map. Fails loudly but usefully."""
    log("· NSE full equity list (archives.nseindia.com, cached to disk)…")
    try:
        uni = ND.load_universe_full_list(cfg.cache_dir, client, log=log)
    except Exception as exc:
        log(f"  ! full list unavailable ({exc})")
        uni = pd.DataFrame(columns=["Symbol", "Company", "Series", "ISIN", "FaceValue",
                                    "DateOfListing"])
    log(f"  → {len(uni)} tradeable equities after excluding weird series")
    log("· Index membership (Nifty 50/100/200/500/Midcap150/Smallcap250)…")
    idx = ND.load_index_membership(cfg.cache_dir, log=log)
    if len(idx):
        log(f"  → {idx['Symbol'].nunique()} symbols carry an index tier "
            f"({idx['IndexTier'].value_counts().to_dict()})")
    if len(uni) and len(idx):
        uni = uni.merge(idx[["Symbol", "Industry", "IndexTier"]], on="Symbol", how="left")
    else:
        uni["Industry"] = np.nan
        uni["IndexTier"] = np.nan
    return uni, idx


# ======================================================================================
# 3. The main pipeline
# ======================================================================================
class Screener:
    """
    One object that owns config, cache, the fetched data and every output artefact, so you
    can re-screen with new parameters in seconds (no re-download) and re-export anything.
    """

    def __init__(self, params: Params, fetch_cfg: FetchConfig, out_dir: str = "out",
                 universe_size: int = 1000, prefer_index_tier: bool = False,
                 use_quotes: bool = False, verbose: bool = True):
        self.p = params
        self.cfg = fetch_cfg
        self.cfg.use_quotes = bool(use_quotes)
        self.out_dir = out_dir
        self.universe_size = int(universe_size)
        self.prefer_index_tier = bool(prefer_index_tier)
        self.verbose = verbose
        os.makedirs(out_dir, exist_ok=True)
        self.log_lines: List[str] = []
        self.client = YahooClient(self.cfg, log=self._log)
        self.universe: Optional[pd.DataFrame] = None
        self.index_tiers: Optional[pd.DataFrame] = None
        self.data: Dict[str, pd.DataFrame] = {}
        self.report: Optional[pd.DataFrame] = None
        self.ranking: Optional[pd.DataFrame] = None
        self.hits: Optional[pd.DataFrame] = None
        self.near: Optional[pd.DataFrame] = None
        self.diag: Optional[pd.DataFrame] = None
        self.finished_at: Optional[str] = None

    # -- logging -----------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        line = str(msg)
        self.log_lines.append(line)
        if self.verbose:
            print(line, flush=True)

    # -- steps -------------------------------------------------------------------------
    def prepare_universe(self) -> "Screener":
        self._log("=" * 78)
        self._log(f"NSE WEEKLY LIQUIDITY-SWEEP SCREENER   {now_ist():%Y-%m-%d %H:%M} IST")
        self._log("=" * 78)
        self.universe, self.index_tiers = build_universe(self.cfg, self.client, self._log)
        return self

    def fetch(self, symbols: Optional[Sequence[str]] = None, progress: bool = True) -> "Screener":
        todo = list(symbols) if symbols is not None else list(self.universe["Symbol"])
        self._log(f"· Fetching {len(todo)} symbols, {self.cfg.history_years}y of daily bars "
                  f"(cache first, {self.cfg.max_workers} workers, "
                  f">={self.cfg.request_sleep}s between calls)…")
        t0 = time.time()
        self.data, self.report = ND.fetch_daily_bulk(todo, self.cfg, self.client,
                                                      progress=progress and self.verbose)
        got = len(self.data)
        attempted = len(self.report) if self.report is not None else 0
        self._log(f"  → usable {got} / {len(todo)} in {time.time()-t0:.0f}s "
                  f"(each symbol accounted for: {attempted == len(todo)})")
        if attempted != len(todo):
            self._log(f"  ! coverage gap: {len(todo)-attempted} symbol(s) have no report row")
        self.coverage = {"asked": len(todo), "reported": attempted, "usable": got}
        self._log("\n" + ND.data_health(self.report, self.cfg))
        ND.save_load_report(self.report, self.data, self.out_dir)
        return self

    def rank_universe(self) -> "Screener":
        """Pick the liquid top-N from everything that has usable history."""
        quotes = None
        if self.cfg.use_quotes:
            try:
                quotes = self.client.quote_batch(list(self.data))
            except Exception as exc:
                self._log(f"  market-cap enrichment skipped ({type(exc).__name__})")
        if quotes is None or len(quotes) == 0:
            quotes = ND.build_quotes_from_meta(self.cfg.cache_dir)
        self.ranking = SE.rank_top(self.data, quotes, top_n=self.universe_size,
                                   index_tiers=self.index_tiers,
                                   prefer_index_tier=self.prefer_index_tier)
        self._log(f"· Ranked universe: {len(self.ranking)} symbols kept "
                  f"(basis={self.ranking['rank_basis'].iloc[0] if len(self.ranking) else '-'})")
        return self

    def screen(self, subset: Optional[Sequence[str]] = None) -> "Screener":
        """Run the sweep detector. subset=None → screen the ranked top-N."""
        if subset is None:
            if self.ranking is None:
                self.rank_universe()
            syms = list(self.ranking["Symbol"])
            self._log(f"· Screening {len(syms)} symbols for completed weekly sweeps "
                      f"(swing_strength={self.p.swing_strength}, lookback={self.p.swing_lookback}w, "
                      f"wick≥{self.p.min_wick_ratio:.0%}, close in upper "
                      f"{self.p.min_close_in_range:.0%}+)…")
        else:
            syms = list(subset)
            self._log(f"· Screening {len(syms)} supplied symbols…")
        data = {s: self.data[s] for s in syms if s in self.data}
        uni = self.universe if self.universe is not None else None
        t0 = time.time()
        self.hits, self.near, self.diag = SE.screen_all(data, self.p, universe=uni)
        self._log(f"  → {len(self.hits)} setups, {len(self.near)} near-misses in "
                  f"{time.time()-t0:.1f}s")
        self.finished_at = now_ist().strftime("%Y-%m-%d %H:%M IST")
        return self

    def screen_all_cached(self) -> "Screener":
        """Ignore the top-N cut and screen literally every cached symbol."""
        return self.screen(subset=list(self.data))

    def set_params(self, **kw) -> "Screener":
        bad = {k: v for k, v in kw.items() if not hasattr(self.p, k)}
        if bad:
            raise KeyError(f"Unknown parameter(s): {list(bad)} — valid ones: "
                           f"{sorted(f for f in self.p.__dataclass_fields__)}")
        self.p = Params(**{**self.p.__dict__, **kw})
        return self

    # -- outputs -----------------------------------------------------------------------
    def summary_frame(self, top: Optional[int] = None) -> pd.DataFrame:
        if self.hits is None or self.hits.empty:
            return pd.DataFrame()
        cols = [c for c in ["Rank", "Symbol", "Company", "Close", "SweepWeek", "BarsSinceSweep",
                            "SweepType", "SweptSwingLow", "SweepLow", "SwingLowMade", "SwingAgeW",
                            "PoolsTaken", "LowPoolRef", "ClosedAboveBy%", "SweepDepth%",
                            "WickRatio%", "CloseInRange%", "Green", "Is52wLow", "VolRatio",
                            "ATR(14w)", "SMA20w", "vsSMA20w%", "Risk%", "Stop", "Target2R",
                            "Target3R", "Resistance26w", "UpsideToRes%", "TurnoverLakh",
                            "Industry", "IndexTier", "Score"] if c in self.hits.columns]
        out = self.hits[cols].copy()
        return out.head(top) if top else out

    def export(self, prefix: str = "sweep") -> Dict[str, str]:
        """Write results.csv / results.xlsx / near_misses.csv / diagnostics.csv + params.json."""
        stamp = now_ist().strftime("%Y-%m-%d")
        paths: Dict[str, str] = {}
        os.makedirs(self.out_dir, exist_ok=True)
        hits = self.summary_frame()
        if not hits.empty:                      # NaN -> empty cell in the exports
            hits = hits.astype(object).where(pd.notna(hits), "")
        p_csv = os.path.join(self.out_dir, f"{prefix}_results_{stamp}.csv")
        (hits if not hits.empty else pd.DataFrame(columns=["Symbol"])).to_csv(p_csv, index=False)
        paths["results_csv"] = p_csv

        p_xlsx = os.path.join(self.out_dir, f"{prefix}_results_{stamp}.xlsx")
        try:
            with pd.ExcelWriter(p_xlsx, engine="openpyxl") as xw:
                (hits if not hits.empty else pd.DataFrame({"note": ["no setups today"]})
                 ).to_excel(xw, sheet_name="Setups", index=False)
                (self.near if self.near is not None and not self.near.empty
                 else pd.DataFrame(columns=["Symbol"])).to_excel(xw, sheet_name="NearMisses",
                                                                  index=False)
                (self.diag if self.diag is not None else pd.DataFrame()).to_excel(
                    xw, sheet_name="ScreenedUniverse", index=False)
                (self.report if self.report is not None else pd.DataFrame()).to_excel(
                    xw, sheet_name="DataQuality", index=False)
                self.params_frame().to_excel(xw, sheet_name="Params", index=False)
                if self.ranking is not None:
                    self.ranking.to_excel(xw, sheet_name="UniverseRanking", index=False)
            paths["results_xlsx"] = p_xlsx
        except Exception as exc:
            self._log(f"  ! xlsx export skipped ({type(exc).__name__}: {exc})")

        if self.near is not None and len(self.near):
            self.near = self.near.rename(columns={"Week": "SweepWeek"})
        if self.near is not None:
            p = os.path.join(self.out_dir, f"{prefix}_near_misses_{stamp}.csv")
            self.near.to_csv(p, index=False)
            paths["near_miss_csv"] = p
        if self.diag is not None:
            p = os.path.join(self.out_dir, f"{prefix}_diagnostics_{stamp}.csv")
            self.diag.to_csv(p, index=False)
            paths["diagnostics_csv"] = p
        p = os.path.join(self.out_dir, f"{prefix}_params_{stamp}.json")
        with open(p, "w") as f:
            json.dump({"generated_ist": self.finished_at, "params": self.p.describe(),
                       "weights": self.p.weights, "fetch": {
                           k: v for k, v in self.cfg.__dict__.items()},
                       "counts": {"universe": int(len(self.universe) if self.universe is not None else 0),
                                  "fetched": len(self.data),
                                  "screened": int(len(self.diag) if self.diag is not None else 0),
                                  "hits": int(len(self.hits) if self.hits is not None else 0)}},
                      f, indent=2, default=str)
        paths["params_json"] = p
        self._log("· Wrote " + ", ".join(os.path.basename(v) for v in paths.values()))
        return paths

    def params_frame(self) -> pd.DataFrame:
        d = {"# of bars since last sweep": self.p.max_sweep_bars_ago}
        d.update(self.p.describe())
        rows = [{"parameter": k, "value": v} for k, v in d.items()]
        rows += [{"parameter": f"weight:{k}", "value": v} for k, v in self.p.weights.items()]
        return pd.DataFrame(rows)

    def telegram_alert(self, bot_token: str, chat_id: str, top_n: int = 10) -> str:
        """Optional push. Keep the token out of the saved notebook: paste at run time."""
        if not bot_token or not chat_id:
            return "skipped (no token/chat id)"
        text = format_message(self.summary_frame(top_n), self.finished_at or "")
        try:
            import requests
            r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                              data={"chat_id": chat_id, "text": text,
                                    "disable_web_page_preview": "true"}, timeout=20)
            return f"telegram {r.status_code}"
        except Exception as exc:
            return f"telegram failed: {type(exc).__name__}"

    # -- analysis ---------------------------------------------------------------------
    def selftest(self) -> pd.DataFrame:
        """
        Detector sanity check on synthetic tapes (no network). Proves the rule works before
        you trust the live run, and shows which filters a real ticker failed.
        """
        import tests_fixtures as TF
        import types
        mod = types.ModuleType("m")
        rows = []
        for name in sorted(d for d in dir(TF) if d.startswith("test_")):
            fn = getattr(TF, name)
            if not callable(fn):
                continue
            try:
                fn()
                rows.append({"check": name, "result": "PASS", "detail": ""})
            except Exception as exc:
                rows.append({"check": name, "result": "FAIL",
                             "detail": f"{type(exc).__name__}: {str(exc)[:160]}"})
        df = pd.DataFrame(rows)
        n = len(df)
        ok = int((df["result"] == "PASS").sum())
        self._log(f"· Self-test: {ok}/{n} checks passed")
        return df

    def diagnose(self, symbols: Sequence[str]) -> pd.DataFrame:
        """
        'Why is X not in my list?' — answers with the exact gate it failed, week by week.
        """
        rows = []
        for sym in symbols:
            daily = self.data.get(str(sym).upper())
            row: Dict[str, object] = {"Symbol": str(sym).upper()}
            if daily is None or len(daily) == 0:
                rep = self.report
                st = "not fetched"
                if rep is not None and "Symbol" in rep.columns:
                    got = rep.loc[rep["Symbol"].astype(str).str.upper() == str(sym).upper(),
                                  "status"]
                    if len(got):
                        st = str(got.iloc[0])
                why = {"no_data": "Yahoo has no bars for this ticker (delisted / renamed / too new)",
                       "throttled": "rate-limited (429) - re-run, the cache makes it cheap",
                       "insufficient_history": "listed too recently for a weekly screen",
                       "illiquid_history": "too many zero-volume days to trust a weekly candle",
                       "stale_data": "history stops well before today",
                       "not_run": "not part of the last fetch (see DataQuality sheet)"}.get(st, st)
                rows.append({**row, "verdict": f"no usable bars - {why}",
                             "weekly_bars": 0, "status": st})
                continue
            wk = SE.resample_weekly(daily)
            row["weekly_bars"] = int(len(wk))
            row["last_week"] = str(pd.Timestamp(wk.index[-1]).date()) if len(wk) else ""
            row["close"] = round(float(wk["Close"].iloc[-1]), 2) if len(wk) else np.nan
            row["turnover_lakh"] = round(SE.turnover_lakh(daily), 1)
            row["locked_weeks_26"] = SE.locked_weeks(wk)
            if len(wk) < self.p.min_weekly_bars:
                row["verdict"] = f"rejected: only {len(wk)} weekly bars (need {self.p.min_weekly_bars})"
                rows.append(row); continue
            i = len(wk) - 1
            lo, hi, cl, op = (float(wk['Low'].iloc[i]), float(wk['High'].iloc[i]),
                              float(wk['Close'].iloc[i]), float(wk['Open'].iloc[i]))
            rng = hi - lo
            if rng <= 0:
                row["verdict"] = "rejected: zero-range last week"
                rows.append(row); continue
            lw = min(op, cl) - lo
            levels = SE._levels_to_sweep(wk, i, self.p)
            row["last_low"] = round(lo, 2)
            row["last_close"] = round(cl, 2)
            row["swing_lows_in_range"] = ", ".join(
                f"{lev:g}@{str(pd.Timestamp(wk.index[j]).date())}" for lev, j, fr in levels[:4]
            ) or "none"
            row["wick_ratio"] = round(lw / rng, 3)
            row["close_in_range"] = round((cl - lo) / rng, 3)
            best = SE.evaluate_bar(wk, i, self.p)
            if best is not None:
                row["verdict"] = f"SWEEP: took {best.level:g}, closed {best.close_above_pct*100:.2f}% above, score {best.score:.0f}"
            else:
                nm = SE.near_miss(wk, self.p, i)
                if nm:
                    row["verdict"] = f"near miss: {nm['FailReason']}"
                else:
                    swept = [lev for lev, j, fr in levels if lo < lev and cl > lev]
                    if not levels:
                        row["verdict"] = "no weekly swing low in lookback window (raise swing_lookback)"
                    elif not swept:
                        row["verdict"] = ("last week did not undercut any swing low"
                                          if lo >= min([l for l, _, _ in levels], default=1e18)
                                          else "undercut the low but CLOSED BELOW it (breakdown, not sweep)")
                    else:
                        row["verdict"] = "swept & reclaimed but failed a quality gate"
            rows.append(row)
        return pd.DataFrame(rows)

    def backtest_rule(self, horizon_weeks: int = 8, r_mult: float = 2.0,
                      max_symbols: Optional[int] = None) -> Tuple[pd.DataFrame, dict]:
        self._log(f"· Backtesting the same rule: {horizon_weeks}w horizon, {r_mult}R target…")
        tr = SE.backtest(self.data, self.p, horizon_weeks=horizon_weeks, r_mult=r_mult,
                         max_symbols=max_symbols)
        summ = SE.summarize_backtest(tr)
        self._log("  → " + json.dumps(summ, default=str))
        if len(tr):
            tr.to_csv(os.path.join(self.out_dir, "backtest_trades.csv"), index=False)
        return tr, summ

    def grid_scan(self, grid: Optional[Dict[str, Sequence[float]]] = None,
                  sample: int = 250, horizon_weeks: int = 8) -> pd.DataFrame:
        """
        Small parameter grid (backtest-driven) so you can see how sensitive the rule is
        instead of trusting one arbitrary setting.
        """
        grid = grid or {"swing_strength": [1, 2, 3], "min_wick_ratio": [0.25, 0.34, 0.45],
                        "max_sweep_bars_ago": [1, 3]}
        keys = list(grid)
        combos = [{}]
        for k in keys:
            combos = [{**c, k: v} for c in combos for v in grid[k]]
        subset = dict(list(self.data.items())[:sample])
        rows = []
        for c in combos:
            p = Params(**{**self.p.__dict__, **c})
            tr = SE.backtest(subset, p, horizon_weeks=horizon_weeks)
            s = SE.summarize_backtest(tr)
            rows.append({**c, **{k: v for k, v in s.items() if k in
                                 ("trades", "win_rate_%", "avg_return_%", "avg_R", "profit_factor")}})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("avg_return_%", ascending=False).reset_index(drop=True)
            df.to_csv(os.path.join(self.out_dir, "param_grid.csv"), index=False)
        return df


# ======================================================================================
# 4. Presentation
# ======================================================================================
def _fmt(v, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}" if abs(v) >= 0.01 or v == 0 else f"{v:.4f}"
    return str(v)


def format_message(hits: pd.DataFrame, stamp: str, max_rows: int = 12) -> str:
    """Compact text summary (terminal or Telegram)."""
    head = f"NSE weekly sweep screen — {stamp}"
    if hits is None or hits.empty:
        return f"{head}\nNo completed weekly sweeps in the screened universe today."
    lines = [head, f"{len(hits)} setup(s); top {min(max_rows, len(hits))}:", ""]
    for _, r in hits.head(max_rows).iterrows():
        lines.append(
            f"{int(r.get('Rank', 0)):>2}. {r['Symbol']:<16} {r['Close']:>9}  "
            f"swept {r['SweptSwingLow']:>9} → low {r['SweepLow']:>9} "
            f"(+{r['ClosedAboveBy%']:.2f}%)  wick {r['WickRatio%']:.0f}%  "
            f"score {r['Score']:.0f}  stop {r['Stop']:>9}  2R {r['Target2R']:>9}")
    lines += ["", "Weekly candles only. Confirm on the chart; risk-manage every entry."]
    return "\n".join(lines)


def style_results(hits: pd.DataFrame, top: int = 40):
    """Colour-graded display frame (works in Colab; degrades to a plain table elsewhere)."""
    if hits is None or hits.empty:
        print("No setups to display.")
        return pd.DataFrame()
    show = hits.head(top).copy()
    num_cols = [c for c in show.columns if pd.api.types.is_numeric_dtype(show[c])]
    display_cols = [c for c in ["Rank", "Symbol", "Company", "Close", "SweepWeek", "BarsSinceSweep",
                                "SweepType", "SweptSwingLow", "SweepLow", "ClosedAboveBy%",
                                "SweepDepth%", "WickRatio%", "CloseInRange%", "Green",
                                "PoolsTaken", "VolRatio", "vsSMA20w%", "Risk%", "Stop",
                                "Target2R", "Target3R", "UpsideToRes%", "TurnoverLakh", "Score",
                                "Industry"]
                    if c in show.columns]
    out = show[display_cols]
    try:
        def _score_css(v):
            try:
                v = float(v)
            except Exception:
                return ""
            if v >= 75:
                return "background:#1b5e20;color:#fff;font-weight:600"
            if v >= 60:
                return "background:#2e7d32;color:#fff"
            if v >= 45:
                return "background:#f9a825;color:#000"
            return "background:#455a64;color:#fff"

        sty = (out.style
               .format(lambda x: "" if pd.isna(x) else (f"{x:.2f}" if isinstance(x, float) else str(x)))
               .map(_score_css, subset=["Score"] if "Score" in out.columns else None)
               .hide(axis="index"))
        if "WickRatio%" in out.columns:
            sty = sty.background_gradient(subset=["WickRatio%"], cmap="Greens")
        if "VolRatio" in out.columns:
            sty = sty.background_gradient(subset=["VolRatio"], cmap="Oranges")
        return sty
    except Exception as exc:
        print(f"(styling unavailable: {type(exc).__name__}) — showing raw table")
        return out


def plot_setup(screener: Screener, symbol: str, weeks: int = 60, engine: str = "plotly",
               height: int = 460):
    """
    Weekly candlestick with the swept liquidity line, the sweep candle highlighted, the
    swing-low markers and the suggested stop / 2R target. Returns a figure object.
    """
    daily = screener.data.get(symbol.upper())
    if daily is None or len(daily) == 0:
        raise KeyError(f"{symbol}: no cached bars (was it fetched?)")
    tail, info = SE.chart_frames(daily, screener.p, weeks_to_show=weeks)
    if tail.empty:
        raise RuntimeError(f"{symbol}: empty weekly tape after resampling")
    x = list(range(len(tail)))
    lbl = [str(pd.Timestamp(d).date()) for d in tail.index]

    if engine == "plotly":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x, open=tail["Open"], high=tail["High"],
                                     low=tail["Low"], close=tail["Close"], name="Weekly",
                                     increasing_line_color="#26a69a",
                                     decreasing_line_color="#ef5350"))
        hit = info.get("hit") or {}
        if hit:
            lev = float(hit["level"])
            fig.add_hline(y=lev, line=dict(color="#ffd54f", width=2, dash="dash"),
                          annotation_text=f"swept swing low {lev:g}", annotation_font_color="#ffd54f")
            p = int(hit.get("sweep_pos", -1))
            if 0 <= p < len(tail):
                fig.add_trace(go.Scatter(x=[p], y=[tail["Low"].iloc[p]], mode="markers",
                                         marker=dict(symbol="triangle-down", size=15,
                                                     color="#ffca28",
                                                     line=dict(width=1, color="#000")),
                                         name="sweep candle", showlegend=True))
        for sw in info.get("swings", []):
            if 0 <= sw["pos"] < len(tail):
                fig.add_trace(go.Scatter(x=[sw["pos"]], y=[sw["low"]], mode="markers",
                                         marker=dict(symbol="circle", size=6,
                                                     color="rgba(66,165,245,0.85)"),
                                         name="swing low", showlegend=False,
                                         hovertext=[f"{sw['date']} low {sw['low']:g}"]))
        close = float(tail["Close"].iloc[-1])
        if hit:
            stop = float(hit["low"]) * 0.995
            risk = close - stop
            if risk > 0:
                fig.add_hline(y=stop, line=dict(color="#ef5350", width=1, dash="dot"),
                              annotation_text="stop", annotation_font_color="#ef5350")
                fig.add_hline(y=close + 2 * risk, line=dict(color="#66bb6a", width=1, dash="dot"),
                              annotation_text="2R", annotation_font_color="#66bb6a")
        fig.update_layout(
            title=(f"{symbol} — weekly | sweep {hit.get('date_sweep','')[:10] if hit else '—'}"
                   f" | score {hit.get('score', 0):.0f}" if hit else f"{symbol} — weekly"),
            xaxis=dict(tickmode="array", tickvals=x, ticktext=lbl, nticks=12,
                       rangeslider=dict(thickness=0)),
            yaxis=dict(title="₹", side="right"), height=height,
            margin=dict(l=8, r=8, t=44, b=8), paper_bgcolor="#111827", plot_bgcolor="#111827",
            font=dict(color="#e5e7eb", size=11), showlegend=False,
            xaxis_rangeslider_visible=False)
        return fig

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, height / 40), facecolor="#111827")
    ax.set_facecolor("#111827")
    for i, (o, h, l, c) in enumerate(zip(tail["Open"], tail["High"], tail["Low"], tail["Close"])):
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=col, lw=1)
        ax.add_patch(plt.Rectangle((i - 0.32, min(o, c)), 0.64, max(abs(c - o), h * 1e-4),
                                   color=col, alpha=0.9))
    hit = info.get("hit") or {}
    if hit:
        ax.axhline(hit["level"], ls="--", color="#ffd54f", lw=1.5,
                   label=f"swept swing low {hit['level']:g}")
        p = int(hit.get("sweep_pos", -1))
        if 0 <= p < len(tail):
            ax.plot(p, tail["Low"].iloc[p], "v", ms=12, color="#ffca28")
    ax.set_xticks(range(0, len(tail), max(1, len(tail) // 10)))
    ax.set_xticklabels([lbl[i] for i in range(0, len(tail), max(1, len(tail) // 10))],
                       rotation=0, fontsize=8, color="#e5e7eb")
    ax.tick_params(colors="#e5e7eb")
    if ax.get_legend_handles_labels()[0]:          # avoid matplotlib's empty-legend warning
        ax.legend(facecolor="#1f2937", labelcolor="#e5e7eb", fontsize=8)
    ax.set_title(f"{symbol} weekly — liquidity sweep", color="#e5e7eb")
    ax.grid(alpha=0.15, color="#6b7280")
    fig.tight_layout()
    return fig


def render_charts(screener: Screener, symbols: Sequence[str], weeks: int = 60,
                  engine: str = "plotly"):
    """Yield (symbol, figure) for each symbol that has data — caller decides how to show."""
    for sym in symbols:
        try:
            yield str(sym), plot_setup(screener, str(sym), weeks=weeks, engine=engine)
        except Exception as exc:
            print(f"  ! {sym}: chart skipped ({type(exc).__name__}: {exc})")


def gallery_html(screener: Screener, symbols: Sequence[str], weeks: int = 55) -> str:
    """Static matplotlib PNGs embedded in one HTML page (nice for saving/sharing)."""
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blocks = []
    for sym in symbols:
        try:
            fig = plot_setup(screener, str(sym), weeks=weeks, engine="mpl")
        except Exception as exc:
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        blocks.append(f'<div style="margin:14px 0"><h3 style="color:#111;margin:4px 0">{sym}'
                      f'</h3><img src="data:image/png;base64,{b64}" style="max-width:100%"></div>')
    return ("<html><body style='font-family:system-ui;background:#fff'>"
            + "".join(blocks) + "</body></html>")


def save_html_gallery(screener: Screener, symbols: Sequence[str], path: str,
                      weeks: int = 55) -> str:
    html = gallery_html(screener, symbols, weeks=weeks)
    with open(path, "w") as f:
        f.write(html)
    return path


# ======================================================================================
# 5. Console helpers
# ======================================================================================
def print_table(df: pd.DataFrame, max_width: int = 200) -> None:
    try:
        with pd.option_context("display.max_columns", None, "display.width", max_width,
                               "display.max_rows", 200, "display.float_format",
                               lambda v: f"{v:,.2f}"):
            print(df.to_string(index=False))
    except Exception:
        print(df.head(40).to_string())


def describe_criteria(p: Params) -> str:
    return f"""BUY SETUP — completed weekly candle only (the live week is excluded, so nothing repaints)
  1. a prior weekly swing low exists  : fractal low with {p.swing_strength} lower/higher bars on each side,
                                        searched up to {p.swing_lookback} weeks back. Fresh lows and the 52-week
                                        low also qualify ({'yes' if p.allow_fresh_low else 'no'}).
  2. that low was TAKEN OUT            : weekly LOW < swing low by ≥ {p.min_depth_pct*100:.2f}% of the level
                                        (and ≤ {p.max_depth_pct*100:.0f}% / {p.max_depth_atr_mult:.1f}× ATR, so a
                                        collapse through it is NOT called a sweep).
  3. price REJECTED back above         : weekly CLOSE > swept level{' + cushion' if p.min_close_above_pct else ''}
                                        (≥ {p.min_close_above_pct*100:.1f}% above, if set).
  4. a PROPER WICK is visible          : lower wick ≥ {p.min_wick_ratio:.0%} of the weekly range,
                                        wick ≥ {p.min_wick_body_mult:.2f}× the body, close in the top
                                        {p.min_close_in_range:.0%} of the range, range ≤ {p.max_range_of_close:.0%} of price.
  5. tradability                       : price ≥ ₹{p.min_price:g}, median daily turnover ≥
                                        ₹{p.min_turnover_lakh:g}L (hard floor at
                                        ₹{p.min_turnover_lakh*p.hard_illiquid_mult:.0f}L), ≤ {p.max_locked_weeks_26}
                                        circuit-locked week in the last 26, ≥ {p.min_weekly_bars} weekly bars.
  6. recency                           : the sweep must be ≤ {p.max_sweep_bars_ago} completed week(s) old.
  Setup Score (0-100) = wick {p.weights['wick']:.0f} + close position {p.weights['close_position']:.0f} + depth {p.weights['depth']:.0f}
                        + volume {p.weights['volume']:.0f} + trend {p.weights['trend']:.0f} + risk-size {p.weights['proximity']:.0f}
                        + structure {p.weights['structure']:.0f} + freshness {p.weights['recency']:.0f}"""