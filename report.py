"""
report.py — turns a finished Screener into one self-contained, visually polished
HTML report (tables + every sweep chart embedded as base64 PNG).

Design constraints:
  * ONE file, no external CSS/JS/images -> renders identically in Gmail's web
    client, in Outlook, and offline from the attachment.
  * Inline styles only. Gmail strips <style> blocks in the message body, so the
    emailed summary must not depend on them.
  * The numbers are taken verbatim from the Screener object. Nothing is
    recomputed here, so the report cannot disagree with the notebook.
"""

from __future__ import annotations

import base64
import html
import io
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

# ---------------------------------------------------------------------------- palette
BG = "#0b1220"
CARD = "#111827"
LINE = "#1f2a3a"
TXT = "#e5e7eb"
MUTE = "#94a3b8"
ACCENT = "#38bdf8"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"

FONT = ("-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',"
        "Arial,sans-serif")
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

# Columns shown in the main table, in order (mirrors the notebook's style_results)
TABLE_COLS = ["Rank", "Symbol", "Company", "Close", "SweepWeek", "SweepType",
              "SweptSwingLow", "SweepLow", "ClosedAboveBy%", "SweepDepth%",
              "WickRatio%", "CloseInRange%", "VolRatio", "vsSMA20w%",
              "Risk%", "Stop", "Target2R", "Target3R", "TurnoverLakh", "Score"]

NEAR_COLS = ["Symbol", "Company", "Close", "SweepType", "SweptSwingLow",
             "SweepLow", "WickRatio%", "CloseInRange%", "ClosedAboveBy%",
             "TurnoverLakh", "FailReason"]


def _esc(v) -> str:
    return html.escape("" if v is None else str(v))


def _cell(v) -> str:
    """Format one table cell the way the notebook's float_format does."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return _esc(v)


def _score_bg(score) -> str:
    """Same thresholds as style_results() in the notebook."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return CARD
    if s >= 75:
        return "#14532d"
    if s >= 60:
        return "#166534"
    if s >= 45:
        return "#78350f"
    return "#334155"


def _kpi(label: str, value: str, colour: str = TXT, sub: str = "") -> str:
    sub_html = (f"<div style='font:400 11px {FONT};color:{MUTE};margin-top:3px'>"
                f"{_esc(sub)}</div>") if sub else ""
    return (
        f"<td style='padding:6px'>"
        f"<div style='background:{CARD};border:1px solid {LINE};border-radius:10px;"
        f"padding:13px 15px'>"
        f"<div style='font:600 10px {FONT};color:{MUTE};letter-spacing:.9px;"
        f"text-transform:uppercase'>{_esc(label)}</div>"
        f"<div style='font:700 25px {FONT};color:{colour};margin-top:5px;"
        f"line-height:1.1'>{_esc(value)}</div>{sub_html}</div></td>")


def _table(df: pd.DataFrame, cols: Sequence[str], max_rows: int,
           highlight_score: bool = True) -> str:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = list(df.columns)[:12]
    body = df.head(max_rows)

    head = "".join(
        f"<th style=\"padding:9px 10px;text-align:{'left' if c in ('Symbol','Company','SweepType','FailReason','Industry') else 'right'};"
        f"font:600 10px {FONT};color:{MUTE};letter-spacing:.6px;text-transform:uppercase;"
        f"border-bottom:2px solid {LINE};white-space:nowrap\">{_esc(c)}</th>"
        for c in cols)

    rows = []
    for i, (_, r) in enumerate(body.iterrows()):
        stripe = "#0e1626" if i % 2 else CARD
        tds = []
        for c in cols:
            align = "left" if c in ("Symbol", "Company", "SweepType",
                                    "FailReason", "Industry") else "right"
            style = (f"padding:8px 10px;text-align:{align};font:400 12px {MONO};"
                     f"color:{TXT};border-bottom:1px solid {LINE};white-space:nowrap")
            if c == "Symbol":
                style = (f"padding:8px 10px;text-align:left;font:700 12.5px {FONT};"
                         f"color:{ACCENT};border-bottom:1px solid {LINE};white-space:nowrap")
            if c == "Company":
                style += ";max-width:190px;overflow:hidden;text-overflow:ellipsis"
            if c == "Score" and highlight_score:
                style = (f"padding:8px 10px;text-align:right;font:700 12.5px {FONT};"
                         f"color:#fff;background:{_score_bg(r.get(c))};"
                         f"border-bottom:1px solid {LINE}")
            tds.append(f"<td style='{style}'>{_cell(r.get(c))}</td>")
        rows.append(f"<tr style='background:{stripe}'>{''.join(tds)}</tr>")

    more = ""
    if len(df) > max_rows:
        more = (f"<div style='font:400 11px {FONT};color:{MUTE};padding:8px 2px'>"
                f"… {len(df) - max_rows} more row(s) — see the attached CSV/XLSX.</div>")

    return (
        f"<div style='overflow-x:auto;border:1px solid {LINE};border-radius:10px'>"
        f"<table cellspacing='0' cellpadding='0' style='width:100%;border-collapse:collapse;"
        f"background:{CARD}'><thead><tr>{head}</tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table></div>{more}")


def _section(title: str, subtitle: str = "") -> str:
    sub = (f"<div style='font:400 12.5px {FONT};color:{MUTE};margin-top:3px'>"
           f"{_esc(subtitle)}</div>") if subtitle else ""
    return (f"<div style='margin:30px 0 12px'>"
            f"<div style='font:700 17px {FONT};color:{TXT}'>{_esc(title)}</div>"
            f"{sub}</div>")


def chart_pngs(sc, symbols: Sequence[str], weeks: int = 70) -> Dict[str, bytes]:
    """
    Render one weekly candlestick PNG per symbol using the notebook's own
    plot_setup() — same swept-low line, same sweep-candle marker, same stop/2R.
    Matplotlib engine so the image embeds in email (plotly needs JS).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import screener as SC

    out: Dict[str, bytes] = {}
    for sym in symbols:
        try:
            fig = SC.plot_setup(sc, str(sym), weeks=weeks, engine="mpl")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            out[str(sym)] = buf.getvalue()
        except Exception as exc:  # one bad symbol must not kill the report
            print(f"  ! chart skipped {sym}: {type(exc).__name__}: {exc}", flush=True)
    return out


def build_html(sc, hits: pd.DataFrame, charts: Dict[str, bytes], stamp: str,
               criteria: str, health: str, stats: Dict[str, object],
               near: Optional[pd.DataFrame] = None,
               cfg_report: Optional[Dict[str, int]] = None) -> str:
    """Assemble the full standalone report."""
    cfg_report = cfg_report or {}
    n_hits = 0 if hits is None else len(hits)
    top = hits.iloc[0] if n_hits else None

    # ---------------------------------------------------------------- header
    head_colour = GREEN if n_hits else AMBER
    verdict = (f"{n_hits} completed weekly sweep{'s' if n_hits != 1 else ''} found"
               if n_hits else "No completed weekly sweep this week")

    parts: List[str] = [
        f"<div style='background:{BG};padding:22px 0'>"
        f"<div style='max-width:1080px;margin:0 auto;padding:0 18px'>",

        # title card
        f"<div style='background:linear-gradient(135deg,#0f172a,#1e293b);"
        f"border:1px solid {LINE};border-left:4px solid {head_colour};"
        f"border-radius:12px;padding:22px 24px'>"
        f"<div style='font:600 11px {FONT};color:{ACCENT};letter-spacing:1.6px;"
        f"text-transform:uppercase'>NSE · Weekly Liquidity-Sweep Screener</div>"
        f"<div style='font:700 26px {FONT};color:{TXT};margin:9px 0 6px'>{_esc(verdict)}</div>"
        f"<div style='font:400 13px {FONT};color:{MUTE}'>Generated {_esc(stamp)} IST · "
        f"swing entries where a <b style='color:{TXT}'>weekly</b> sweep of a swing low is "
        f"<b style='color:{TXT}'>complete</b></div></div>",
    ]

    # ---------------------------------------------------------------- KPIs
    kpis = [
        _kpi("Setups", str(n_hits), GREEN if n_hits else MUTE, "passed every gate"),
        _kpi("Screened", f"{stats.get('screened', 0):,}", TXT,
             f"of {stats.get('universe', 0):,} NSE equities"),
        _kpi("Near misses", f"{0 if near is None else len(near)}", AMBER,
             "one gate away"),
        _kpi("Top score",
             f"{float(top['Score']):.0f}" if (top is not None and pd.notna(top.get("Score"))) else "—",
             TXT,
             str(top["Symbol"]) if top is not None else ""),
    ]
    parts.append("<table cellspacing='0' cellpadding='0' style='width:100%;margin:14px 0 0'>"
                 f"<tr>{''.join(kpis)}</tr></table>")

    # data-quality strip
    parts.append(
        f"<div style='background:{CARD};border:1px solid {LINE};border-radius:10px;"
        f"padding:11px 15px;margin-top:12px;font:400 12px {FONT};color:{MUTE}'>"
        f"<b style='color:{TXT}'>Data</b> · {stats.get('fetched', 0):,} symbols with usable "
        f"history · {stats.get('http_calls', 0):,} HTTP calls "
        f"({stats.get('calls_per_symbol', 0):.2f}/symbol) · "
        f"{stats.get('rate_limited', 0)} × 429 · {stats.get('retries', 0)} retries · "
        f"fetch {stats.get('fetch_secs', 0):.0f}s, screen {stats.get('screen_secs', 0):.1f}s"
        f"</div>")

    # ---------------------------------------------------------------- setups table
    if n_hits:
        parts.append(_section(
            "The setups",
            "Ranked by Setup Score. Stop = 0.5% under the sweep low; targets are 2R/3R."))
        parts.append(_table(hits, TABLE_COLS, cfg_report.get("table_rows", 40)))
    else:
        parts.append(
            f"<div style='background:{CARD};border:1px solid {LINE};border-left:4px solid {AMBER};"
            f"border-radius:10px;padding:18px 20px;margin-top:22px;font:400 14px {FONT};"
            f"color:{TXT}'>No stock in the screened universe printed a completed weekly "
            f"sweep-and-reclaim this week. That is a normal, honest answer — the pattern is "
            f"not supposed to appear every week. The near-misses below show what the filter "
            f"rejected and why.</div>")

    # ---------------------------------------------------------------- charts
    if charts:
        parts.append(_section(
            "Charts — the sweep, marked",
            "Weekly candles. Dashed amber line = the swept swing low. "
            "▼ = the sweep candle. Dotted red = stop, dotted green = 2R target."))
        for sym, png in charts.items():
            b64 = base64.b64encode(png).decode()
            row = None
            if n_hits:
                match = hits[hits["Symbol"].astype(str) == sym]
                if len(match):
                    row = match.iloc[0]
            meta = ""
            if row is not None:
                meta = (
                    f"<div style='font:400 12px {MONO};color:{MUTE};margin:2px 0 9px'>"
                    f"close <b style='color:{TXT}'>{_cell(row.get('Close'))}</b> · "
                    f"swept <b style='color:{AMBER}'>{_cell(row.get('SweptSwingLow'))}</b> · "
                    f"sweep low {_cell(row.get('SweepLow'))} · "
                    f"wick {_cell(row.get('WickRatio%'))}% · "
                    f"stop <b style='color:{RED}'>{_cell(row.get('Stop'))}</b> · "
                    f"2R <b style='color:{GREEN}'>{_cell(row.get('Target2R'))}</b> · "
                    f"score <b style='color:{TXT}'>{_cell(row.get('Score'))}</b></div>")
            parts.append(
                f"<div style='background:{CARD};border:1px solid {LINE};border-radius:10px;"
                f"padding:15px;margin-bottom:14px'>"
                f"<div style='font:700 15px {FONT};color:{ACCENT}'>{_esc(sym)}</div>{meta}"
                f"<img src='data:image/png;base64,{b64}' "
                f"style='width:100%;max-width:1020px;border-radius:6px;display:block'/></div>")

    # ---------------------------------------------------------------- near misses
    if near is not None and len(near):
        parts.append(_section(
            "Near misses — one gate away",
            "Useful for calibration: these swept a low but failed exactly one rule."))
        parts.append(_table(near, NEAR_COLS, cfg_report.get("near_miss_rows", 15),
                            highlight_score=False))

    # ---------------------------------------------------------------- criteria + health
    parts.append(_section("The rule that was applied",
                          "Verbatim from the screener's own describe_criteria()."))
    parts.append(
        f"<pre style='background:{CARD};border:1px solid {LINE};border-radius:10px;"
        f"padding:15px 17px;font:400 11.5px {MONO};color:{MUTE};white-space:pre-wrap;"
        f"overflow-x:auto;margin:0'>{_esc(criteria)}</pre>")

    if health:
        parts.append(_section("Data health"))
        parts.append(
            f"<pre style='background:{CARD};border:1px solid {LINE};border-radius:10px;"
            f"padding:15px 17px;font:400 11.5px {MONO};color:{MUTE};white-space:pre-wrap;"
            f"overflow-x:auto;margin:0'>{_esc(health)}</pre>")

    # ---------------------------------------------------------------- footer
    parts.append(
        f"<div style='margin:30px 0 6px;padding:16px 18px;background:{CARD};"
        f"border:1px solid {LINE};border-radius:10px;font:400 12px {FONT};color:{MUTE}'>"
        f"<b style='color:{AMBER}'>⚠ Not investment advice.</b> This is a screening tool that "
        f"finds a liquidity-raid-and-reclaim pattern on completed weekly candles. It has no "
        f"opinion on fundamentals, news, sector rotation or market regime. Confirm on the "
        f"chart and risk-manage every entry — you own the risk.<br><br>"
        f"Generated automatically by <b style='color:{TXT}'>weekly-sweep</b> on GitHub Actions "
        f"from NSE_Weekly_Liquidity_Sweep_Screener.ipynb · every Saturday 19:30 IST."
        f"</div></div></div>")

    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>NSE Weekly Sweep · {_esc(stamp)}</title></head>"
            f"<body style='margin:0;background:{BG}'>" + "".join(parts) + "</body></html>")
