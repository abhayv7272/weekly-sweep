# weekly-sweep

Automated **NSE Weekly Liquidity-Sweep Screener** — runs the Colab notebook's
engine on GitHub Actions every **Saturday 19:30 IST** and emails a
visually-complete HTML report (tables + every sweep chart) to `REPORT_TO`.

## What it screens

NSE equities whose latest **completed weekly candle** swept (raided) a swing low
and **closed back above it with a proper rejection wick**. Old swing lows *and*
fresh lows (window low / 52-week low) both qualify. Universe: all NSE equities
with usable history, ranked by liquidity, **top 1000 screened**.

The live week is always excluded, so nothing repaints.

## Layout

| Path | What it is |
|---|---|
| `notebook/NSE_Weekly_Liquidity_Sweep_Screener.ipynb` | the source of truth |
| `wlsweep/` | the four engine modules, **extracted verbatim** from the notebook |
| `extract_engine.py` | regenerates `wlsweep/` from the notebook |
| `config.py` | `CONFIG` copied verbatim from notebook cell §2 |
| `run_sweep.py` | headless runner — same pipeline, same order as the notebook |
| `report.py` | builds the standalone HTML report (charts embedded as base64) |
| `emailer.py` | Gmail SMTP delivery |
| `.github/workflows/weekly-sweep.yml` | the Saturday schedule |

### Why the result matches Colab

`run_sweep.py` contains **no screening logic**. It imports the same
`nse_data` / `sweep_engine` / `screener` modules the notebook writes at runtime,
feeds them the same `CONFIG`, and calls the same methods in the same order
(`prepare_universe → fetch → rank_universe → screen → summary_frame → export`).
The report only formats `sc.summary_frame()`; it never recomputes a number.

The engine's own **49-check self-test suite** runs before every screen — if the
detector is broken the job aborts instead of emailing you wrong numbers.

## Setup

Three repository secrets (Settings → Secrets and variables → Actions):

| Secret | Value |
|---|---|
| `GMAIL_USER` | the Gmail address that sends |
| `GMAIL_APP_PASSWORD` | a Google [App Password](https://myaccount.google.com/apppasswords) (not your login password) |
| `REPORT_TO` | where the report lands |

## Running it by hand

Actions → **Weekly Sweep Report** → *Run workflow*.

- `limit` — screen only the first N symbols (`40` for a ~2-minute smoke test; `0` = full universe)
- `send_email` — untick to write the report as an artifact without emailing

Locally:

```bash
pip install -r requirements.txt
python run_sweep.py --no-email          # writes out/
python run_sweep.py --limit 40          # quick test, sends email
```

## Tuning the screen

Edit `config.py`. Useful knobs:

- `max_sweep_bars_ago` — `1` = only the week just closed; raise to `3–4` for a wider net
- `min_score` — `0` shows everything, `55` keeps only A-grade setups
- `universe_size` — how many liquid names get screened
- `min_wick_ratio`, `min_close_in_range` — how strict the rejection must be

## Cost & runtime

First run downloads ~2 300 symbols (≈6–8 min). The cache is stored between runs
via `actions/cache`, so later Saturdays only top up the new bars.

---

⚠️ **Not investment advice.** A screening tool that finds a
liquidity-raid-and-reclaim pattern. It has no opinion on fundamentals, news,
sector rotation or market regime. You own the risk.
