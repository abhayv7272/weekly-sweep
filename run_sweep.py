#!/usr/bin/env python3
"""
run_sweep.py — headless runner for the NSE Weekly Liquidity-Sweep Screener.

It executes EXACTLY the notebook's pipeline, in the notebook's order, with the
notebook's CONFIG:

    §0  import the engine modules (wlsweep/, extracted verbatim from the .ipynb)
    §2  build Params / FetchConfig from CONFIG
    §3  prepare_universe() -> fetch() -> data_health()
    §4  rank_universe()
    §5  screen() -> summary_frame()     <- the numbers you see in Colab
    §6  charts for the top setups       <- plot_setup(), same markings
    §8  export() CSV / XLSX / JSON
    §9  self-test suite (49 checks)
    §11 email the report

No screening logic lives in this file; it only orchestrates and reports, so the
emailed result cannot drift from the notebook's result.

Usage:
    python run_sweep.py                 # run + email
    python run_sweep.py --no-email      # run, write out/ only
    python run_sweep.py --limit 40      # smoke test on a small universe
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(ROOT, "wlsweep")
OUT_DIR = os.path.join(ROOT, "out")
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

import pandas as pd  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

# ---- the engine, exactly as the notebook imports it (cell §1 tail) ------------------
import nse_data as ND           # noqa: E402
import sweep_engine as SE       # noqa: E402
import screener as SC           # noqa: E402
from nse_data import FetchConfig  # noqa: E402
from sweep_engine import Params   # noqa: E402
from screener import Screener, describe_criteria  # noqa: E402

import report as RP             # noqa: E402
from config import CONFIG, REPORT  # noqa: E402


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true", help="write out/ but do not send")
    ap.add_argument("--limit", type=int, default=0,
                    help="screen only the first N symbols (smoke test)")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    log("=" * 78)
    log(f"NSE WEEKLY LIQUIDITY-SWEEP SCREENER — automated run")
    log(f"started {SC.now_ist():%Y-%m-%d %H:%M:%S} IST")
    log("=" * 78)

    # ---------------------------------------------------------------- §9.1 self-test
    if not args.skip_selftest:
        log("\n[1/7] self-test suite (synthetic tapes, no network)")
        try:
            import subprocess
            r = subprocess.run([sys.executable, "-m", "pytest", "tests_fixtures.py", "-q"],
                               cwd=PKG_DIR, capture_output=True, text=True, timeout=600)
            log("      " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
            if r.returncode != 0:
                log("      ! self-test FAILED — aborting, the engine is not trustworthy")
                log(r.stdout[-3000:])
                return 2
        except Exception as exc:
            log(f"      self-test skipped ({type(exc).__name__}: {exc})")

    # ---------------------------------------------------------------- §2 parameters
    log("\n[2/7] parameters")
    cfg_params = {k: v for k, v in CONFIG.items() if k in Params.__dataclass_fields__}
    cfg_fetch = {k: v for k, v in CONFIG.items() if k in FetchConfig.__dataclass_fields__}
    fetch_cfg = FetchConfig(**cfg_fetch)
    params = Params(**cfg_params)
    criteria = describe_criteria(params)
    log("      " + criteria.splitlines()[0])

    # cache lives next to the repo so Actions can restore it between runs
    fetch_cfg.cache_dir = os.path.join(ROOT, CONFIG.get("cache_dir", "cache_nse"))

    sc = Screener(params, fetch_cfg, out_dir=OUT_DIR,
                  universe_size=CONFIG["universe_size"],
                  prefer_index_tier=CONFIG["prefer_index_tier"],
                  use_quotes=CONFIG["use_quotes"], verbose=True)

    # ---------------------------------------------------------------- §3 universe + fetch
    log("\n[3/7] universe & fetch")
    sc.prepare_universe()
    symbols = list(sc.universe["Symbol"])
    if args.limit:
        symbols = symbols[:args.limit]
        log(f"      --limit {args.limit}: screening {len(symbols)} symbol(s) only")
    t0 = time.time()
    sc.fetch(symbols, progress=False)
    fetch_secs = time.time() - t0

    # ---------------------------------------------------------------- §4 ranking
    log("\n[4/7] ranking")
    sc.rank_universe()
    log(f"      ranked {len(sc.ranking)} symbols")

    # ---------------------------------------------------------------- §5 screen
    log("\n[5/7] screening")
    t0 = time.time()
    sc.screen()
    screen_secs = time.time() - t0
    hits = sc.summary_frame()
    near = sc.near
    log(f"      setups {len(hits)} · near-misses {0 if near is None else len(near)} · "
        f"screened {0 if sc.diag is None else len(sc.diag)} in {screen_secs:.1f}s")
    if len(hits):
        log("\n" + SC.format_message(hits, sc.finished_at or ""))

    # ---------------------------------------------------------------- §8 exports
    log("\n[6/7] exports & charts")
    paths = sc.export("nse")
    for k, v in paths.items():
        log(f"      {k:16s} {os.path.basename(v)}  ({os.path.getsize(v):,} B)")

    # charts: the setups if any, else the top ranked names for context (notebook §6/§23)
    if len(hits):
        gal = list(hits["Symbol"].head(REPORT["gallery_symbols"]))
    elif sc.ranking is not None and len(sc.ranking):
        gal = list(sc.ranking["Symbol"].head(6))
    else:
        gal = []
    charts = RP.chart_pngs(sc, gal, weeks=REPORT["gallery_weeks"]) if gal else {}
    log(f"      charts rendered: {len(charts)}")

    # the interactive plotly gallery, exactly as notebook §23 saves it
    gallery_path = None
    if gal:
        try:
            gallery_path = SC.save_html_gallery(
                sc, gal, os.path.join(OUT_DIR, "sweep_gallery.html"),
                weeks=REPORT["gallery_weeks"])
        except Exception as exc:
            log(f"      gallery skipped ({type(exc).__name__}: {exc})")

    # ---------------------------------------------------------------- the report
    stamp = SC.now_ist().strftime("%Y-%m-%d %H:%M")
    stats = {
        "universe": len(sc.universe) if sc.universe is not None else 0,
        "fetched": len(sc.data),
        "screened": len(sc.diag) if sc.diag is not None else 0,
        "http_calls": sc.client.stats.get("http_calls", 0),
        "calls_per_symbol": (sc.client.stats.get("http_calls", 0) /
                             max(1, len(sc.report) if sc.report is not None else 1)),
        "rate_limited": sc.client.stats.get("rate_limited_429", 0),
        "retries": sc.client.stats.get("retries", 0),
        "fetch_secs": fetch_secs,
        "screen_secs": screen_secs,
    }
    try:
        health = ND.data_health(sc.report, sc.cfg)
    except Exception:
        health = ""

    html = RP.build_html(sc, hits, charts, stamp, criteria, health, stats,
                         near=near, cfg_report=REPORT)
    report_path = os.path.join(OUT_DIR, f"weekly_sweep_report_{SC.now_ist():%Y-%m-%d}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"      report           {os.path.basename(report_path)}  "
        f"({os.path.getsize(report_path):,} B)")

    # ---------------------------------------------------------------- §11 email
    log("\n[7/7] email")
    if args.no_email:
        log("      --no-email: skipped")
    else:
        import emailer
        atts = [(os.path.basename(report_path), html.encode("utf-8"), "html")]
        for key in ("results_csv", "results_xlsx", "near_miss_csv"):
            p = paths.get(key)
            if p and os.path.exists(p) and os.path.getsize(p) < 12_000_000:
                sub = "csv" if p.endswith(".csv") else "octet-stream"
                atts.append((os.path.basename(p), open(p, "rb").read(), sub))
        if gallery_path and os.path.getsize(gallery_path) < 12_000_000:
            atts.append(("sweep_gallery.html", open(gallery_path, "rb").read(), "html"))
        for sym, png in charts.items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sym)
            atts.append((f"chart_{safe}.png", png, "png"))

        n = len(hits)
        subject = (f"NSE Weekly Sweep · {n} setup{'s' if n != 1 else ''} · "
                   f"{SC.now_ist():%d %b %Y}") if n else \
                  f"NSE Weekly Sweep · no setups · {SC.now_ist():%d %b %Y}"
        emailer.send_report(subject, html, atts)

    log(f"\ndone in {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
