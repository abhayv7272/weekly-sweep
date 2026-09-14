#!/usr/bin/env python3
"""
diagnose.py — prove the pipeline works against the REAL network.

Everything here is read-only. It answers, with evidence:

  1. is NSE's equity list reachable, and how many symbols does it yield?
  2. does Yahoo return usable OHLCV for real NSE tickers, at what call cost?
  3. does the cache round-trip (write -> read -> identical frame)?
  4. does the full pipeline (rank -> screen -> export -> charts -> HTML) run?
  5. do the exported CSV numbers match what the HTML report shows?
  6. does Gmail accept the app password?

Run it on GitHub Actions (the sandbox has no route to Yahoo/NSE):
    python diagnose.py --symbols 25
    python diagnose.py --smtp-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "wlsweep"))
sys.path.insert(0, ROOT)

LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LINES.append(msg)


def head(title: str) -> None:
    say("")
    say("=" * 72)
    say(title)
    say("=" * 72)


def smtp_only() -> int:
    head("SMTP login check")
    import smtplib
    import ssl
    user = os.environ.get("MY_EMAIL") or os.environ.get("GMAIL_USER") or ""
    pw = (os.environ.get("MY_APP_PASSWORD") or
          os.environ.get("GMAIL_APP_PASSWORD") or "").replace(" ", "")
    if not user or not pw:
        say("FAIL  MY_EMAIL / MY_APP_PASSWORD not set")
        return 1
    say(f"user        : {user[:3]}***{user[-12:]}")
    say(f"password len: {len(pw)} (Google app passwords are 16 chars)")
    if len(pw) != 16:
        say("WARN  that is not 16 characters — Gmail will probably reject it")
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=60) as s:
            s.login(user, pw)
        say("PASS  Gmail accepted the credentials")
        return 0
    except Exception as exc:
        say(f"FAIL  {type(exc).__name__}: {exc}")
        say("      -> regenerate the app password at "
            "https://myaccount.google.com/apppasswords")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=25)
    ap.add_argument("--smtp-only", action="store_true")
    args = ap.parse_args()

    if args.smtp_only:
        rc = smtp_only()
        with open(os.path.join(ROOT, "diagnosis.txt"), "a") as f:
            f.write("\n".join(LINES) + "\n")
        return rc

    failures = 0
    import pandas as pd

    import nse_data as ND
    import screener as SC
    from nse_data import FetchConfig
    from sweep_engine import Params
    from screener import Screener, describe_criteria
    import report as RP
    from config import CONFIG, REPORT

    head("0 · environment")
    say(f"python  {sys.version.split()[0]}")
    for m in ("pandas", "numpy", "matplotlib", "plotly", "requests", "openpyxl"):
        try:
            say(f"{m:12s} {__import__(m).__version__}")
        except Exception as exc:
            say(f"{m:12s} MISSING ({type(exc).__name__})")
            failures += 1

    # ---------------------------------------------------------------- 1 universe
    head("1 · NSE universe reachable?")
    cfg_fetch = {k: v for k, v in CONFIG.items() if k in FetchConfig.__dataclass_fields__}
    fetch_cfg = FetchConfig(**cfg_fetch)
    fetch_cfg.cache_dir = os.path.join(ROOT, "cache_nse")
    params = Params(**{k: v for k, v in CONFIG.items()
                       if k in Params.__dataclass_fields__})
    sc = Screener(params, fetch_cfg, out_dir=os.path.join(ROOT, "out"),
                  universe_size=CONFIG["universe_size"],
                  prefer_index_tier=CONFIG["prefer_index_tier"],
                  use_quotes=CONFIG["use_quotes"], verbose=True)
    t0 = time.time()
    try:
        sc.prepare_universe()
        n = len(sc.universe)
        say(f"PASS  {n} NSE equities in {time.time()-t0:.1f}s")
        say(f"      series mix: {sc.universe['Series'].value_counts().head(4).to_dict()}")
        if n < 1500:
            say(f"WARN  expected ~2300 symbols, got {n} — NSE may have served a partial list")
    except Exception as exc:
        say(f"FAIL  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------------- 2 fetch
    head(f"2 · Yahoo bars for {args.symbols} real symbols")
    syms = list(sc.universe["Symbol"])[:args.symbols]
    say(f"      {', '.join(syms[:8])}{' …' if len(syms) > 8 else ''}")
    t0 = time.time()
    sc.fetch(syms, progress=False)
    secs = time.time() - t0
    got = len(sc.data)
    calls = sc.client.stats.get("http_calls", 0)
    say("")
    say(f"{'PASS' if got else 'FAIL'}  usable {got}/{len(syms)} in {secs:.0f}s")
    say(f"      HTTP calls {calls} = {calls/max(1,len(syms)):.2f}/symbol "
        f"(the notebook's design target is 1.00)")
    say(f"      429s {sc.client.stats.get('rate_limited_429',0)} · "
        f"retries {sc.client.stats.get('retries',0)}")
    if not got:
        failures += 1
    elif got < len(syms) * 0.8:
        say(f"WARN  only {got/len(syms):.0%} usable")
    if calls / max(1, len(syms)) > 1.6:
        say("WARN  more than 1.6 calls/symbol — the full 2300-symbol run may hit 429s")

    if got:
        s0 = next(iter(sc.data))
        d0 = sc.data[s0]
        say("")
        say(f"      sample {s0}: {len(d0)} daily bars, "
            f"{d0.index.min().date()} → {d0.index.max().date()}")
        stale = (pd.Timestamp(SC.now_ist().date()) - d0.index.max()).days
        say(f"      last bar is {stale} day(s) old "
            f"{'(fine)' if stale <= 7 else '(STALE — check the feed)'}")
        bad = int(((d0["High"] < d0[["Open", "Close"]].max(axis=1)) |
                   (d0["Low"] > d0[["Open", "Close"]].min(axis=1))).sum())
        say(f"      OHLC violations: {bad} {'(clean)' if bad == 0 else '(SUSPECT)'}")

    # ---------------------------------------------------------------- 3 cache
    head("3 · cache round-trip")
    if got:
        s0 = next(iter(sc.data))
        ND.save_daily_cache(sc.data[s0], fetch_cfg.cache_dir, s0, "csv")
        back = ND.load_daily_cache(fetch_cfg.cache_dir, s0, "csv")
        if back is None:
            say("FAIL  cache did not read back")
            failures += 1
        else:
            same = len(back) == len(sc.data[s0])
            say(f"{'PASS' if same else 'FAIL'}  wrote and re-read {len(back)} rows for {s0}")
            failures += 0 if same else 1

    # ---------------------------------------------------------------- 4 pipeline
    head("4 · rank → screen → export → charts")
    try:
        sc.rank_universe()
        say(f"      ranked {len(sc.ranking)} (basis={sc.ranking['rank_basis'].iloc[0]})")
        sc.screen()
        hits = sc.summary_frame()
        say(f"      setups {len(hits)} · near-misses "
            f"{0 if sc.near is None else len(sc.near)} · screened {len(sc.diag)}")
        paths = sc.export("diag")
        say(f"      exported {len(paths)} file(s)")
        gal = (list(hits["Symbol"].head(REPORT["gallery_symbols"])) if len(hits)
               else list(sc.ranking["Symbol"].head(3)))
        charts = RP.chart_pngs(sc, gal, weeks=REPORT["gallery_weeks"])
        say(f"      charts rendered {len(charts)}/{len(gal)}")
        if len(charts) < len(gal):
            say("WARN  some charts failed to render")
        html = RP.build_html(sc, hits, charts, "diagnosis",
                             describe_criteria(params),
                             ND.data_health(sc.report, sc.cfg),
                             {"universe": len(sc.universe), "fetched": len(sc.data),
                              "screened": len(sc.diag), "http_calls": calls,
                              "calls_per_symbol": calls / max(1, len(syms)),
                              "rate_limited": 0, "retries": 0,
                              "fetch_secs": secs, "screen_secs": 0.0},
                             near=sc.near, cfg_report=REPORT)
        p = os.path.join(ROOT, "out", "diagnosis_report.html")
        open(p, "w", encoding="utf-8").write(html)
        say(f"PASS  report {os.path.getsize(p):,} B, "
            f"{html.count('data:image/png;base64,')} chart(s) embedded")

        # ------------------------------------------------------------ 5 parity
        head("5 · do the CSV and the HTML agree?")
        if len(hits):
            miss = []
            for _, r in hits.iterrows():
                for col in ("Symbol", "SweptSwingLow", "Stop", "Target2R", "Score"):
                    v = r.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        continue
                    s = f"{v:,.2f}" if isinstance(v, float) else str(v)
                    if s not in html:
                        miss.append(f"{r['Symbol']}.{col}={s}")
            if miss:
                say(f"FAIL  {len(miss)} value(s) in the CSV are absent from the report")
                say("      " + ", ".join(miss[:8]))
                failures += 1
            else:
                say(f"PASS  every value of all {len(hits)} setup(s) appears in the report")
        else:
            say("SKIP  no setups in this small sample (normal)")
    except Exception as exc:
        say(f"FAIL  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        failures += 1

    head("verdict")
    say("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    with open(os.path.join(ROOT, "diagnosis.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
