"""
nse_data.py — resilient market-data layer for the NSE Weekly Liquidity Sweep Screener.

Design goals (this file exists because "data fetching" is the failure mode):
  * Multi-source universe resolution (NSE archive CSV -> local cache -> pasted list).
  * Multi-source OHLCV: Yahoo chart API (primary, plain requests) -> Yahoo weekly
    fallback -> local CSV cache -> NSE archived daily bhavcopy (deep fallback).
  * Politely rate-limited, retrying with exponential backoff + jitter, host rotation,
    adaptive backpressure when the API pushes 429/5xx, and per-symbol failure
    classification so the screener can *prove* data quality instead of hoping.
  * On-disk cache (gzip CSV) so a Colab re-run costs ~0 network calls.

No paid APIs, no key, no scraping of the anti-bot NSE HTML pages.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional, nicer progress bar in Colab
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")

_UAS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
)

# Series that are unusable / not tradeable as normal swing candidates
_EXCLUDE_SERIES = {"BE", "BL", "B1", "BC", "Z", "Z9", "8P", "G", "GC", "GW", "WQ", "V", "V1"}
_INCLUDE_SERIES_DEFAULT = {"EQ", "A", "AQ", "AQ1000", "BE", "SM", "SS", "MT", "M1", "MZ"}

CSV_SECTIONS = ["Company Name", "Symbol", "Series", "ISIN", "Industry"]

# Symbols Yahoo no longer serves under the NSE ticker (demergers/renames). The screener
# reports them explicitly instead of silently dropping them; extend this if you hit more.
YAHOO_ALIASES: Dict[str, str] = {
    "TATAMOTORS": "TATAMOTORS.NS",      # delisted from Yahoo post demerger -> reported, not guessed
}


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class FetchConfig:
    """All knobs that control *how* data is fetched (not what is screened)."""

    cache_dir: str = "cache_nse"
    history_years: int = 6            # daily history requested per symbol
    min_weekly_bars: int = 104        # 2 years of weekly bars, else "insufficient history"
    max_fresh_lag_days: int = 10      # last bar older than this => data_quality flag
    request_sleep: float = 0.10       # min gap between two HTTP calls (per worker)
    transport: str = "auto"           # auto | curl_cffi | requests | yfinance
    throttle_pause: float = 45.0      # park time after repeated 429s
    final_pass: bool = True           # retry throttled symbols once, slowly, on another transport
    quote_chunk: int = 100            # symbols per batch-quote call (auto-shrinks on pushback)
    max_retries: int = 5
    backoff_base: float = 1.7
    backoff_cap: float = 40.0
    timeout: float = 25.0
    max_workers: int = 3
    cache_format: str = "csv"         # 'csv' (gzip); parquet also works if pyarrow is present
    use_quotes: bool = False          # optional: try Yahoo batch-quotes for true market cap
    quote_max_calls: int = 6          # ...but never hammer: Yahoo throttles this endpoint hard
    alias_lookup: bool = True         # use the YAHOO_ALIASES map for delisted/renamed tickers
    fetch_reuse_cache_off: bool = False  # True = ignore local cache and refetch everything
    bhavcopy_fallback: bool = False   # deep rescue: pull NSE bhavcopy zips (one file per DAY)
    bhavcopy_max_days: int = 40       # cap: each zip is the whole market, so keep this small


# --------------------------------------------------------------------------------------
# Shared backoff state (one place decides how polite we are, and it adapts)
# --------------------------------------------------------------------------------------
class BackoffState:
    """
    Process-wide throttle state shared by every worker thread.

    * enforces a minimum gap between consecutive HTTP calls
    * doubles the gap after a 429 (and decays it back after successes)
    * pauses everyone when a retry window is pending
    """

    def __init__(self, base_interval: float = 0.25, max_interval: float = 12.0):
        self.base = float(base_interval)
        self.max = float(max_interval)
        self.mult = 1.0
        self.cooldown_until = 0.0
        self._lock = threading.Lock()

    @property
    def interval(self) -> float:
        return min(self.max, self.base * self.mult)

    def schedule(self, secs: float) -> None:
        with self._lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + float(secs))

    def bump(self, factor: float = 2.0) -> None:
        with self._lock:
            self.mult = min(self.max / max(self.base, 1e-6), max(1.0, self.mult * factor))

    def relax(self) -> None:
        with self._lock:
            self.mult = max(1.0, self.mult * 0.7)

    def acquire(self) -> None:
        """Block until it is polite to issue the next request."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait_for = max(0.0, self.cooldown_until - now)
                gap = self.interval
            if wait_for > 0:
                time.sleep(min(wait_for, 10.0))
                continue
            with self._lock:
                now = time.monotonic()
                if now - getattr(self, "_last", 0.0) >= self.interval:
                    self._last = now
                    return
            time.sleep(min(gap, 0.05))


# --------------------------------------------------------------------------------------
# Session / HTTP plumbing
# --------------------------------------------------------------------------------------
class YahooClient:
    """Thin, polite, self-healing wrapper around the public Yahoo endpoints."""

    RETRYABLE = (429, 500, 502, 503, 504, 520, 521, 522)

    def __init__(self, cfg: FetchConfig, log: Optional[Callable[[str], None]] = None,
                 transport: Optional[str] = None):
        self.cfg = cfg
        self.log = log or (lambda m: None)
        self.transport = transport or (cfg.transport or "auto")
        if self.transport == "auto":
            self.transport = "curl_cffi" if _have_curl_cffi() else "requests"
        if self.transport not in ("curl_cffi", "requests", "yfinance"):
            self.transport = "requests"
        self.state = BackoffState(base_interval=max(0.02, cfg.request_sleep))
        self.stats: Dict[str, float] = {
            "http_calls": 0, "retries": 0, "rate_limited_429": 0, "server_errors": 0,
            "not_found": 0, "parse_errors": 0, "network_errors": 0, "backoff_seconds": 0.0,
        }
        self._local = threading.local()
        self._crumb_lock = threading.Lock()
        self._crumb: Optional[str] = None
        self._crumb_attempted = False

    # -- transport --------------------------------------------------------------------
    def _session(self):
        sess = getattr(self._local, "sess", None)
        if sess is None:
            if self.transport == "curl_cffi":
                try:
                    from curl_cffi import requests as cr          # type: ignore
                    sess = cr.Session(impersonate="chrome")
                    self._local.sess = sess
                    return sess
                except Exception:
                    self.transport = "requests"
            import requests
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": random.choice(_UAS),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://finance.yahoo.com/",
                "Connection": "keep-alive",
            })
            self._local.sess = sess
        return sess

    def _raw_get(self, url: str, params: dict):
        """One HTTP GET through the active transport, returning a response-like object."""
        if self.transport == "curl_cffi":
            return self._session().get(url, params=params, timeout=self.cfg.timeout)
        if self.transport == "yfinance":
            return _YfResponse(self._yf_fetch(url, params))
        return self._session().get(url, params=params, timeout=self.cfg.timeout,
                                   headers={"Accept-Encoding": "gzip, deflate"})

    def _yf_fetch(self, url: str, params: dict):
        """Last-resort transport: let yfinance do the request (it has its own cookie/crumb logic)."""
        import yfinance as yf  # type: ignore
        sym = url.split("/chart/")[-1].split("?")[0]
        rng = str(params.get("range") or "max")
        ivl = str(params.get("interval") or "1d")
        if "period1" in params:
            h = yf.Ticker(sym).history(start=str(pd.to_datetime(int(params["period1"]), unit="s").date()),
                                       end=str(pd.to_datetime(int(params["period2"]), unit="s").date()),
                                       interval=ivl, auto_adjust=False, actions=False)
        else:
            h = yf.Ticker(sym).history(period=rng, interval=ivl, auto_adjust=False, actions=False)
        if h is None or h.empty:
            return None
        idx = pd.to_datetime(h.index)
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None) if getattr(idx, "tz", None) else idx
        return {"chart": {"result": [{
            "timestamp": [int(t.timestamp()) for t in idx],
            "meta": {"symbol": sym, "longName": sym},
            "indicators": {"quote": [{
                "open": [float(v) if pd.notna(v) else None for v in h["Open"]],
                "high": [float(v) if pd.notna(v) else None for v in h["High"]],
                "low": [float(v) if pd.notna(v) else None for v in h["Low"]],
                "close": [float(v) if pd.notna(v) else None for v in h["Close"]],
                "volume": [float(v) if pd.notna(v) else 0.0 for v in h.get("Volume", [])]
                if "Volume" in h.columns else [0.0] * len(h)}]}}]}}

    def throttled_recently(self, within: float = 20.0) -> bool:
        """True if a 429 landed in the last `within` seconds (used to skip optional calls)."""
        return (time.monotonic() - getattr(self, "_last_429_ts", 0.0)) < within

    def switch_transport(self, name: str) -> None:
        self.transport = name
        self._local = threading.local()
        self.log(f"    transport -> {name}")

    def _record(self, key: str, n: float = 1) -> None:
        with self._crumb_lock:
            self.stats[key] = self.stats.get(key, 0) + n

    # -- raw GET with retry / backoff / host rotation ----------------------------------
    def _get_json(self, path: str, params: dict, *, expect_crumb: bool = False,
                  max_attempts: Optional[int] = None) -> Optional[dict]:
        attempts = int(max_attempts if max_attempts is not None else self.cfg.max_retries)
        last_err = "unknown"
        for n in range(attempts):
            host = YAHOO_HOSTS[n % len(YAHOO_HOSTS)]
            url = f"https://{host}{path}"
            p = dict(params)
            crumb_used = False
            if expect_crumb:
                crumb = self.get_crumb()
                if crumb:
                    p["crumb"] = crumb
                    crumb_used = True
            self.state.acquire()
            self._record("http_calls")
            try:
                r = self._raw_get(url, p)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                self._record("network_errors")
                self.state.bump(1.5)
                delay = min(self.cfg.backoff_cap, self.cfg.backoff_base ** n)
                delay *= (0.6 + 0.8 * random.random())
                self.state.schedule(delay)
                self._record("backoff_seconds", delay)
                continue

            code = getattr(r, "status_code", 200)
            if code == 200:
                self._429_run = 0
                self.state.relax()
                try:
                    return r.json()
                except Exception as exc:
                    last_err = f"json decode: {exc}"
                    self._record("parse_errors")
                    continue
            if code == 404:
                self._record("not_found")
                return None
            if code == 429:
                # the only thing that works here is to slow down, never to hammer harder
                self._record("rate_limited_429")
                self._last_429_ts = time.monotonic()
                self._429_run = getattr(self, "_429_run", 0) + 1
                last_err = "HTTP 429 (rate limited)"
                try:
                    wait = float(r.headers.get("Retry-After") or 0.0)
                except Exception:
                    wait = 0.0
                delay = max(wait, min(self.cfg.backoff_cap,
                                      self.cfg.backoff_base ** (n + 1)) * 1.5)
                self.state.bump(2.0)
                self.state.schedule(delay)
                self._record("backoff_seconds", delay)
                if self._429_run >= 6:                # everybody is being throttled
                    self.state.schedule(max(delay, self.cfg.throttle_pause))
                    self._429_run = 0
                continue
            if code in (401, 403) and expect_crumb and not crumb_used:
                with self._crumb_lock:
                    self._crumb, self._crumb_attempted = None, False
                last_err = f"HTTP {code} (crumb refresh requested)"
                continue
            if code in self.RETRYABLE:
                self._record("server_errors")
                last_err = f"HTTP {code}"
                delay = min(self.cfg.backoff_cap, self.cfg.backoff_base ** n)
                self.state.bump(1.3)
                self.state.schedule(delay)
                self._record("backoff_seconds", delay)
                continue
            last_err = f"HTTP {code}: {r.text[:120]}"
            self._record("parse_errors")
            return None
        self.log(f"    ! giving up on {path}: {last_err}")
        return None

    # -- crumb (only the batch quote endpoint needs it) --------------------------------
    def get_crumb(self) -> Optional[str]:
        with self._crumb_lock:
            if self._crumb_attempted:
                return self._crumb
            self._crumb_attempted = True
        sess = self._session()
        crumb = None
        try:
            try:
                sess.get("https://finance.yahoo.com/markets/stocks/", timeout=15)
            except Exception:
                pass
            self.state.acquire()
            r = sess.get(f"https://{YAHOO_HOSTS[0]}/v1/test/getcrumb", timeout=15,
                         headers={"Accept": "text/plain, */*"})
            if r.status_code == 200:
                c = r.text.strip()
                crumb = c if _looks_like_crumb(c) else None
        except Exception:
            crumb = None
        with self._crumb_lock:
            self._crumb = crumb
        return crumb

    # -- chart data --------------------------------------------------------------------
    def chart(self, yahoo_symbol: str, period: str = "max", interval: str = "1d",
              start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Daily/weekly bars only (one HTTP call)."""
        df, _ = self.chart_with_meta(yahoo_symbol, period=period, interval=interval,
                                      start=start, end=end)
        return df

    def chart_with_meta(self, yahoo_symbol: str, period: str = "max",
                        interval: str = "1d", start: Optional[str] = None,
                        end: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
        """
        ONE HTTP call, both the bars and the little meta block Yahoo sends for free
        (name, last price, 52w range). Keeping this at exactly one request per symbol is
        the whole reason the bulk fetch survives Yahoo's throttling.
        """
        params: dict = {"interval": interval, "includePrePost": "false",
                        "events": "div,split"}
        if start and end:
            params["period1"] = _to_epoch(start)
            params["period2"] = _to_epoch(end)
        else:
            params["range"] = period
        js = self._get_json(f"/v8/finance/chart/{yahoo_symbol}", params,
                            max_attempts=max(3, int(self.cfg.max_retries)))
        meta: dict = {}
        try:
            meta = (((js or {}).get("chart") or {}).get("result") or [{}])[0].get("meta", {}) or {}
        except Exception:
            meta = {}
        return parse_chart_json(js), meta

    def quote_batch(self, yahoo_symbols: Sequence[str]) -> pd.DataFrame:
        """
        Market cap / price / average volume for many symbols. Uses the batch quote endpoint
        (needs cookie+crumb) and *shrinks the chunk* whenever Yahoo pushes back, so a rate
        limit degrades throughput instead of destroying the ranking.
        """
        fields = ("symbol,marketCap,regularMarketPrice,sharesOutstanding,fiftyTwoWeekLow,"
                  "fiftyTwoWeekHigh,twoHundredDayAverage,averageDailyVolume3Month,"
                  "averageDailyVolume10Day,longName,shortName,regularMarketVolume,currency,"
                  "exchange,priceToBook,trailingPE,firstTradeDateMilliseconds")
        syms = [x if x.endswith(".NS") else f"{x}.NS" for x in yahoo_symbols]
        rows: List[dict] = []
        chunk = max(25, int(self.cfg.quote_chunk))
        # this endpoint is throttled hard by Yahoo, so: big chunk, few calls, quit early
        n_chunks = -(-len(syms) // chunk)
        allowed = int(self.cfg.quote_max_calls)
        if n_chunks > allowed:                       # spread the budget over bigger chunks
            chunk = max(120, -(-len(syms) // allowed))
            n_chunks = -(-len(syms) // chunk)
            if n_chunks > allowed:
                syms = syms[: chunk * allowed]        # and simply rank the rest by turnover
                n_chunks = allowed
        i = 0
        fails = 0
        while i < len(syms):
            batch = syms[i:i + chunk]
            js = self._get_json("/v7/finance/quote",
                                {"symbols": ",".join(batch), "fields": fields,
                                 "formatted": "false"},
                                expect_crumb=True, max_attempts=2)
            res = ((js or {}).get("quoteResponse") or {}).get("result")
            if res is None:
                fails += 1
                if fails >= 2:
                    self.log("    quote endpoint is throttling us -> falling back to "
                             "turnover-based ranking (no market cap). This does not affect "
                             "the screen, only the 'top 1000' ordering.")
                    break
                i += chunk                            # skip this slice rather than loop forever
                continue
            fails = 0
            rows.extend(res)
            i += chunk
        if not rows:
            return pd.DataFrame(columns=["Symbol", "Name", "MarketCap", "Close", "AvgVolume3M"])
        q = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="first")
        q["Symbol"] = q["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False)
        for src, dst in (("marketCap", "MarketCap"), ("regularMarketPrice", "Close"),
                         ("averageDailyVolume3Month", "AvgVolume3M"),
                         ("sharesOutstanding", "SharesOutstanding"),
                         ("fiftyTwoWeekLow", "Low52W"), ("fiftyTwoWeekHigh", "High52W"),
                         ("twoHundredDayAverage", "SMA200D"), ("priceToBook", "Pb"),
                         ("trailingPE", "PE")):
            q[dst] = pd.to_numeric(q.get(src), errors="coerce") if src in q.columns else np.nan
        if "longName" in q.columns or "shortName" in q.columns:
            ln = q.get("longName")
            sn = q.get("shortName")
            q["Name"] = ln if ln is None else ln.fillna(sn)
        else:
            q["Name"] = ""
        cols = ["Symbol", "Name", "MarketCap", "Close", "AvgVolume3M", "SharesOutstanding",
                "Low52W", "High52W", "SMA200D", "Pb", "PE"]
        return q.reindex(columns=cols)

    def lookup(self, query: str, limit: int = 10) -> pd.DataFrame:
        js = self._get_json("/v1/finance/lookup",
                            {"query": query, "type": "equity", "count": limit},
                            max_attempts=2)
        res = ((js or {}).get("finance") or {}).get("result") or []
        rows = []
        for block in res:
            for qq in block.get("quotes", []) or []:
                rows.append({"yahoo_symbol": qq.get("symbol"),
                             "description": qq.get("shortname") or qq.get("longname"),
                             "exch": qq.get("exchDisp") or qq.get("exch")})
        return pd.DataFrame(rows)


class _YfResponse:
    """Minimal requests.Response stand-in for the yfinance transport."""

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200 if payload else 404
        self.headers = {}
        self.text = "" if payload else "no data"

    def json(self):
        if self._payload is None:
            raise ValueError("empty payload")
        return self._payload


def _have_curl_cffi() -> bool:
    try:
        import curl_cffi  # noqa: F401
        return True
    except Exception:
        return False


def _looks_like_crumb(c: str) -> bool:
    return bool(c) and len(c) < 32 and re.fullmatch(r"[A-Za-z0-9;/=\-\._%+]+", c) is not None


def _to_epoch(date_str: str) -> str:
    d = pd.to_datetime(date_str).tz_localize(IST)
    return str(int(d.timestamp()))


# --------------------------------------------------------------------------------------
# Yahoo JSON parsing
# --------------------------------------------------------------------------------------
def parse_chart_json(js: Optional[dict]) -> pd.DataFrame:
    """chart JSON -> DataFrame[Open,High,Low,Close,Volume] indexed by IST date (daily)."""
    empty = pd.DataFrame(columns=OHLCV_COLS)
    if not js:
        return empty
    chart = js.get("chart") or {}
    result = chart.get("result")
    if not result:
        return empty
    r = result[0]
    ts = r.get("timestamp")
    if not ts:
        return empty
    quote = (r.get("indicators") or {}).get("quote") or [{}]
    q = quote[0] or {}
    adj = (r.get("indicators") or {}).get("adjclose") or []
    df = pd.DataFrame({
        "Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
        "Close": q.get("close"), "Volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s", utc=True))
    if adj and adj[0].get("adjclose") is not None:
        # keep raw close (screener trades price, not adjusted returns) but store ratio
        df["AdjClose"] = pd.Series(adj[0]["adjclose"], index=df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(IST)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    # A bar is only usable if OHLC are all finite and High >= Low > 0.
    ok = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    ok &= df["High"] >= df["Low"]
    ok &= df["Low"] > 0
    df = df[ok].astype(float)
    return df.dropna(subset=["Close"])


# --------------------------------------------------------------------------------------
# Universe resolution
# --------------------------------------------------------------------------------------
def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), dtype=str, skipinitialspace=True)


def load_universe_full_list(cache_dir: str, client: Optional[YahooClient] = None,
                            allow_network: bool = True, log: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """
    All NSE-listed equities with Series / ISIN / face value.
    Sources tried in order: archives.nseindia.com, nseindia.com mirror, cached file,
    bundled sample. Returns columns: Symbol, Company, Series, ISIN, FaceValue, DateOfListing.
    """
    log = log or (lambda m: None)
    try:
        os.makedirs(cache_dir, exist_ok=True)     # read-only / disconnected Drive must not
    except Exception as exc:                      # turn "use the cache" into a hard crash
        log(f"  ! cannot create {cache_dir} ({type(exc).__name__}) - trying read-only use")
    cache_fp = os.path.join(cache_dir, "universe_equity_l.csv")
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www.nsearchives.co.in/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    raw = None
    if allow_network:
        for u in urls:
            try:
                sess = (client or YahooClient(FetchConfig()))._session()
                r = sess.get(u, timeout=25)
                if r.status_code == 200 and len(r.content) > 500:
                    raw = r.content
                    with open(cache_fp, "wb") as f:
                        f.write(raw)
                    log(f"  universe: downloaded {u.split('/')[2]} ({len(raw)//1024} KB)")
                    break
            except Exception as exc:
                log(f"  universe: {u.split('/')[2]} failed ({type(exc).__name__})")
    if raw is None and os.path.exists(cache_fp):
        log(f"  universe: using cached copy {cache_fp}")
        raw = open(cache_fp, "rb").read()
    if raw is None:
        raise RuntimeError(
            "Could not obtain the NSE equity list (network blocked and no cache). "
            "Download EQUITY_L.csv from archives.nseindia.com/content/equities/EQUITY_L.csv "
            "on your laptop, upload it to the Colab as 'cache_nse/universe_equity_l.csv', "
            "or paste your own symbol list (UNIVERSE['pasted_symbols'])."
        )
    df = _read_csv_bytes(raw)
    ren = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "symbol":
            ren[c] = "Symbol"
        elif "name" in cl:
            ren[c] = "Company"
        elif cl == "series":
            ren[c] = "Series"
        elif "isin" in cl:
            ren[c] = "ISIN"
        elif "face" in cl:
            ren[c] = "FaceValue"
        elif "listing" in cl:
            ren[c] = "DateOfListing"
    df = df.rename(columns=ren)
    if "Symbol" not in df.columns:
        raise RuntimeError(f"Unexpected EQUITY_L.csv header: {list(df.columns)[:8]}")
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    for c in ("Series", "Company", "ISIN", "FaceValue", "DateOfListing"):
        if c not in df.columns:
            df[c] = np.nan
    df = df.dropna(subset=["Symbol"]).drop_duplicates(subset=["Symbol"])
    if "Series" in df.columns:
        df["Series"] = df["Series"].fillna("").astype(str).str.strip()
        df = df[~df["Series"].str.upper().isin(_EXCLUDE_SERIES)]
    return df[["Symbol", "Company", "Series", "ISIN", "FaceValue", "DateOfListing"]].reset_index(drop=True)


def load_industry_map(cache_dir: str, allow_network: bool = True,
                      log: Optional[Callable[[str], None]] = None) -> pd.Series:
    """Symbol -> Industry, from the Nifty 500 constituent file (best-effort, non fatal)."""
    log = log or (lambda m: None)
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, "universe_nifty500_industry.csv")
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    raw = None
    if allow_network:
        for u in urls:
            try:
                import requests
                r = requests.get(u, timeout=25, headers={"User-Agent": _UAS[0]})
                if r.status_code == 200 and len(r.content) > 200:
                    raw = r.content
                    with open(fp, "wb") as f:
                        f.write(raw)
                    break
            except Exception:
                pass
    if raw is None and os.path.exists(fp):
        raw = open(fp, "rb").read()
    if raw is None:
        log("  industry map unavailable (offline) — continuing without sector column")
        return pd.Series(dtype=object)
    try:
        df = _read_csv_bytes(raw).rename(columns=lambda c: c.strip())
        if "Symbol" in df.columns and "Industry" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
            return df.drop_duplicates("Symbol").set_index("Symbol")["Industry"]
    except Exception:
        pass
    return pd.Series(dtype=object)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------
def _cache_path(cache_dir: str, symbol: str, interval: str, fmt: str) -> str:
    # NSE tickers really do contain "&" (M&M, S&P...) which is not filename-safe. Map it to a
    # token that cannot collide with a *different* real symbol, so two tickers never share one
    # cache file (read and write go through here, so they can never disagree).
    sym = re.sub(r"\s+", "", str(symbol).upper())
    safe = sym.replace("&", "_AMP_")
    safe = re.sub(r"[^A-Z0-9_.-]", "_", safe)
    return os.path.join(cache_dir, f"{safe}.{interval}.{fmt}")


def save_daily_cache(df: pd.DataFrame, cache_dir: str, symbol: str, fmt: str = "csv") -> None:
    if df is None or df.empty:
        return
    os.makedirs(cache_dir, exist_ok=True)
    fp = _cache_path(cache_dir, symbol, "1d", fmt)
    try:
        if fmt == "parquet":
            df.to_parquet(fp)
        else:
            with gzip.open(fp + ".gz", "wt") as f:
                df.to_csv(f)
    except Exception:
        pass


def load_daily_cache(cache_dir: str, symbol: str, fmt: str = "csv") -> Optional[pd.DataFrame]:
    fp = _cache_path(cache_dir, symbol, "1d", fmt)
    try:
        if fmt == "parquet" and os.path.exists(fp):
            return pd.read_parquet(fp)
        if os.path.exists(fp + ".gz"):
            with gzip.open(fp + ".gz", "rt") as f:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
            for c in OHLCV_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
    except Exception:
        return None
    return None


def save_meta(meta: dict, cache_dir: str, symbol: str) -> None:
    """Persist the free per-symbol meta (name, price, 52w range) next to the bars."""
    if not meta:
        return
    try:
        d = os.path.join(cache_dir, "meta")
        os.makedirs(d, exist_ok=True)
        pd.DataFrame([{
            "Symbol": str(meta.get("symbol", symbol)).replace(".NS", "").upper(),
            "Name": meta.get("longName") or meta.get("shortName") or "",
            "Close": meta.get("regularMarketPrice"),
            "High52W": meta.get("fiftyTwoWeekHigh"),
            "Low52W": meta.get("fiftyTwoWeekLow"),
            "PrevClose": meta.get("chartPreviousClose"),
            "FirstTrade": pd.to_datetime(meta.get("firstTradeDate"), unit="s")
            if meta.get("firstTradeDate") else pd.NaT,
            "LastQuote": pd.to_datetime(meta.get("regularMarketTime"), unit="s", utc=True)
            if meta.get("regularMarketTime") else pd.NaT,
        }]).to_csv(os.path.join(d, f"{symbol.upper()}.meta.csv"), index=False)
    except Exception:
        pass


def build_quotes_from_meta(cache_dir: str) -> pd.DataFrame:
    """
    Rebuild a quotes table from the meta files written during the bulk fetch. Zero extra
    network calls, and it gives the notebook company names + 52-week context for free.
    """
    d = os.path.join(cache_dir, "meta")
    if not os.path.isdir(d):
        return pd.DataFrame(columns=["Symbol", "Name", "Close", "High52W", "Low52W", "MarketCap"])
    frames = []
    for f in os.scandir(d):
        if f.name.endswith(".meta.csv"):
            try:
                frames.append(pd.read_csv(f.path))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Name", "Close", "High52W", "Low52W", "MarketCap"])
    q = pd.concat(frames, ignore_index=True).drop_duplicates("Symbol")
    if "MarketCap" not in q.columns:
        q["MarketCap"] = np.nan
    return q


# --------------------------------------------------------------------------------------
# Index membership (used to sanity-check the 'top 1000' ordering)
# --------------------------------------------------------------------------------------
INDEX_FILES = {
    "Nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "Nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NiftyMidcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NiftySmallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}


def load_index_membership(cache_dir: str, allow_network: bool = True,
                          log: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """
    Symbol -> best index tier found in the NSE archive (all free, no cookies, no bot wall).
    Returns columns [Symbol, IndexTier, Industry]. Never raises: a missing file is not fatal.
    """
    log = log or (lambda m: None)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    rank = {k: n for n, k in enumerate(INDEX_FILES)}
    best: Dict[str, Tuple[int, str, str]] = {}
    import requests
    for name, url in INDEX_FILES.items():
        fp = os.path.join(cache_dir, f"index_{name}.csv")
        raw = None
        if allow_network:
            for attempt in range(2):
                try:
                    r = requests.get(url, timeout=25, headers={"User-Agent": _UAS[attempt % len(_UAS)]})
                    if r.status_code == 200 and len(r.content) > 150:
                        raw = r.content
                        with open(fp, "wb") as f:
                            f.write(raw)
                        break
                    if r.status_code == 404:
                        break
                except Exception:
                    time.sleep(1.0)
        if raw is None and os.path.exists(fp):
            raw = open(fp, "rb").read()
        if raw is None:
            continue
        try:
            df = _read_csv_bytes(raw).rename(columns=lambda c: str(c).strip())
            if "Symbol" not in df.columns:
                continue
            ind = df["Industry"] if "Industry" in df.columns else pd.Series([""] * len(df))
            for sym, industry in zip(df["Symbol"].astype(str), ind.astype(str)):
                sym = sym.strip().upper()
                if not sym:
                    continue
                cur = best.get(sym)
                if cur is None or rank[name] < cur[0]:
                    best[sym] = (rank[name], name, industry)
        except Exception as exc:
            log(f"  index {name}: parse failed ({type(exc).__name__})")
    if not best:
        log("  index membership unavailable (offline?) - ranking on turnover only")
        return pd.DataFrame(columns=["Symbol", "IndexTier", "Industry"])
    out = pd.DataFrame([{"Symbol": k, "IndexTier": v[1], "Industry": v[2], "_r": v[0]}
                        for k, v in best.items()]).sort_values("_r").drop(columns="_r")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Data-quality gate
# --------------------------------------------------------------------------------------
def assess_daily(df: pd.DataFrame, *, min_weekly_bars: int, max_fresh_lag_days: int,
                  today: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, List[str], dict]:
    """Clean a daily frame + return (clean_df, list_of_issues, metrics)."""
    issues: List[str] = []
    metrics: dict = {}
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS), ["no_data"], metrics
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame(columns=OHLCV_COLS), ["bad_index"], metrics
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(IST).tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = [c for c in OHLCV_COLS if c in df.columns]
    df = df[keep]

    # ---- repair, do not delete -------------------------------------------------------
    # Yahoo intermittently ships bars whose High/Low ignore Open/Close (seen on NORBTEAEXP:
    # Open 6.35, High 6.30). Dropping them loses real history and can hide a sweep; taking the
    # max/min of the four fields recovers a bar that is true to the tape.
    four = df[["Open", "High", "Low", "Close"]]
    h_true = four.max(axis=1)
    l_true = four[["Open", "Low", "Close"]].min(axis=1)
    viol = (df["High"] < h_true - 1e-9) | (df["Low"] > l_true + 1e-9) | (df["High"] < df["Low"])
    n_repaired = int(viol.sum())
    if n_repaired:
        df.loc[viol, "High"] = h_true[viol]
        df.loc[viol, "Low"] = l_true[viol]
        issues.append(f"repaired {n_repaired} inconsistent OHLC bar(s)")
    metrics["ohlc_repaired"] = n_repaired
    # only genuinely unusable rows get dropped
    bad = (df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
           | (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1))
    if bad.any():
        df = df[~bad.to_numpy()]
        issues.append(f"dropped {int(bad.sum())} corrupt bar(s)")
        if len(df) < 2:
            return pd.DataFrame(columns=OHLCV_COLS), ["no_data"], metrics

    # Yahoo returns a row for every NSE *holiday* too: zero volume, close carried forward.
    # Those are not "sessions" - left in, they poison the volume ratio, the turnover
    # estimate, the week's Day count and the no-trade detector. Drop the pure carry-forward
    # rows; keep a zero-volume day if the price actually moved (a real halt is informative).
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    prev_close = df["Close"].shift(1)
    carry = (vol <= 0) & ((df["Close"] - prev_close).abs() <= 1e-9)
    zero_vol = int((vol <= 0).sum())
    metrics["zero_volume_days"] = zero_vol
    metrics["holiday_rows_dropped"] = int(carry.sum())
    if carry.any():
        df = df[~carry.to_numpy()]
        if len(df) < 2:
            return pd.DataFrame(columns=OHLCV_COLS), ["no_data"], metrics
    if len(df) >= 60:
        frac = zero_vol / len(df)
        if frac > 0.25:
            issues.append(f"low liquidity: {frac:.0%} zero-volume days")

    # ---- corporate-action / bad-feed artefacts -------------------------------------
    # Splits ARE pre-adjusted by Yahoo (verified: WIPRO 2:1 on 2024-12-03 shows no price
    # jump), but a demerger or a bad feed can leave a huge one-day step that invents or
    # destroys a swing low. Flag it so the user can judge the affected symbol.
    r = df["Close"].pct_change()
    worst = float(r.abs().max()) if len(r) else 0.0
    big = r[r.abs() > 0.35]
    metrics["max_1d_move"] = round(float(worst) * 100, 2) if np.isfinite(worst) else None
    metrics["gap_days"] = int(len(big))
    if len(big) >= 3:
        issues.append(f"{len(big)} unexplained >35% one-day moves (demerger/bonus or bad feed) "
                      f"max {metrics['max_1d_move']}%")
    elif len(big):
        issues.append(f"{len(big)} >35% one-day move(s) - check for corporate action")

    n_weeks = max(0, int(math.floor(len(df) / 5.0)))
    metrics["daily_rows"] = int(len(df))
    metrics["first_day"] = str(df.index[0].date()) if len(df) else None
    metrics["last_day"] = str(df.index[-1].date()) if len(df) else None
    if n_weeks < min_weekly_bars:
        issues.append(f"only ~{n_weeks} weekly bars (need {min_weekly_bars})")

    ref = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(IST).date())
    lag = int((ref - df.index[-1]).days)
    metrics["staleness_days"] = lag
    if lag > max(20, max_fresh_lag_days * 3):
        issues.append(f"very stale data ({lag} days)")
    elif lag > max_fresh_lag_days:
        issues.append(f"stale by {lag} days")

    if "Close" in df.columns and len(df) > 30:
        metrics["price"] = float(df["Close"].iloc[-1])
        if metrics["price"] <= 0:
            issues.append("non-positive close")
    return df, issues, metrics


def bhavcopy_available() -> bool:
    """Cheap check that the NSE daily-archive bucket is reachable (no scraping, no cookies)."""
    import requests
    d = (datetime.now(IST) - timedelta(days=4)).strftime("%Y%m%d")
    u = f"https://archives.nseindia.com/historical/contract%20notes/SME_SqRdBhav_{d}.zip"
    try:
        r = requests.head(u, timeout=10, headers={"User-Agent": _UAS[0]}, allow_redirects=True)
        return r.status_code in (200, 404)  # 404 = bucket alive, wrong date is fine
    except Exception:
        return False


def fetch_bhavcopy_days(start: pd.Timestamp, end: pd.Timestamp, max_days: int = 60) -> pd.DataFrame:
    """
    Deep fallback: concatenated daily bhavcopies from the NSE archive bucket.
    Returns columns: Symbol, date, Open/High/Low/Close/Volume. Never raises.
    """
    import requests
    rows = []
    d = pd.Timestamp(start).date()
    stop = pd.Timestamp(end).date()
    if (stop - d).days > max_days:
        stop = d + timedelta(days=max_days)
    while d <= stop:
        if d.weekday() < 5:  # Mon-Fri only (holidays just 404)
            for kind in ("SqRdBhav", "Sq8Bhav"):
                url = f"https://archives.nseindia.com/historical/contract%20notes/{kind}_{d.strftime('%Y%m%d')}.zip"
                try:
                    r = requests.get(url, timeout=25, headers={"User-Agent": _UAS[0]})
                    if r.status_code != 200:
                        continue
                    import zipfile
                    zf = zipfile.ZipFile(io.BytesIO(r.content))
                    name = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                    if not name:
                        continue
                    txt = zf.read(name[0]).decode("latin1")
                    first = txt.splitlines()[0] if txt else ""
                    if first.strip().upper().startswith("symbol"):
                        df = pd.read_csv(io.StringIO(txt))
                        cols = {c.strip().lower(): c for c in df.columns}
                        need = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "price": "Close"}
                        out = pd.DataFrame()
                        out["Symbol"] = df[cols["symbol"]].astype(str).str.upper().str.strip()
                        for src, dst in need.items():
                            if src in cols:
                                out[dst] = pd.to_numeric(df[cols[src]], errors="coerce")
                        if "avgprice" in cols or "volume" in cols:
                            out["Volume"] = pd.to_numeric(df[cols.get("volume", cols.get("avgprice"))], errors="coerce")
                        out["date"] = pd.Timestamp(d)
                        rows.append(out)
                    break
                except Exception:
                    continue
        d = d + timedelta(days=1)
    if not rows:
        return pd.DataFrame()
    allb = pd.concat(rows, ignore_index=True)
    allb = allb.dropna(subset=["Open", "High", "Low", "Close"]).drop_duplicates(["Symbol", "date"])
    allb["date"] = pd.to_datetime(allb["date"])
    return allb.set_index(["Symbol", "date"]).sort_index()


def bhavcopy_to_daily(allb: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Slice one symbol out of a MultiIndex(Symbol, date) bhavcopy frame."""
    if allb is None or allb.empty or not isinstance(allb.index, pd.MultiIndex):
        return pd.DataFrame(columns=OHLCV_COLS)
    try:
        if symbol not in allb.index.get_level_values(0):
            return pd.DataFrame(columns=OHLCV_COLS)
        sub = allb.xs(symbol, level=0).copy()
        sub.index = pd.to_datetime(sub.index)
        cols = [c for c in OHLCV_COLS if c in sub.columns]
        return sub[cols].apply(pd.to_numeric, errors="coerce").dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLS)


# --------------------------------------------------------------------------------------
# The big fetcher
# --------------------------------------------------------------------------------------
def fetch_daily_bulk(symbols: Sequence[str], cfg: FetchConfig, client: YahooClient,
                     progress: bool = True,
                     on_progress: Optional[Callable[[dict], None]] = None
                     ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Fetch + validate daily OHLCV for `symbols`, with a local cache in front of the network.

    Order of attempts per symbol:
        1. local cache file            (free, reproducible, survives runtime restarts)
        2. Yahoo daily bars            (primary)
        3. Yahoo weekly bars           (works when daily history is embargoed/short)
        4. Yahoo lookup -> renamed ticker retry (delistings/spin-offs, e.g. TATAMOTORS)

    Returns
    -------
    data   : dict symbol -> cleaned daily DataFrame (symbols that passed the quality gate)
    report : per-symbol status / row count / first / last / source / issue
    """
    data: Dict[str, pd.DataFrame] = {}
    report_rows: Dict[str, dict] = {}
    lock = threading.Lock()
    todo = list(dict.fromkeys(str(x).upper() for x in symbols if str(x).strip()))
    period = f"{max(2, int(cfg.history_years))}y"

    def _yahoo_symbol(sym: str) -> str:
        return sym if sym.endswith(".NS") else f"{sym}.NS"

    def _load_meta(sym: str) -> Optional[dict]:
        fp = os.path.join(cfg.cache_dir, "meta", f"{sym}.meta.csv")
        if os.path.exists(fp):
            try:
                return pd.read_csv(fp).iloc[-1].to_dict()
            except Exception:
                return None
        return None

    def _one(sym: str) -> dict:
        t0 = time.time()
        rec = {"Symbol": sym, "status": "ok", "rows": 0, "source": "-", "issues": "",
               "elapsed": 0.0, "first": "", "last": "", "alias": ""}
        df, used, meta = None, "cache", {}
        if not cfg.fetch_reuse_cache_off:
            df = load_daily_cache(cfg.cache_dir, sym, cfg.cache_format)
        if df is None or df.empty:
            # --- exactly one network call in the normal path ---
            df, meta = client.chart_with_meta(_yahoo_symbol(sym), period=period, interval="1d")
            used = "yahoo"
            if meta:
                save_meta(meta, cfg.cache_dir, sym)
        if (df is None or df.empty) and not client.throttled_recently():
            # only when Yahoo *really* has no daily bars (young listing / embargoed), never
            # after a throttle - a second call on a 429 just digs the hole deeper
            w, wmeta = client.chart_with_meta(_yahoo_symbol(sym), period=period, interval="1wk")
            if not w.empty:
                df, used = w, "yahoo_weekly"
                if wmeta:
                    save_meta(wmeta, cfg.cache_dir, sym)
        if (df is None or df.empty) and cfg.alias_lookup:
            alias = YAHOO_ALIASES.get(sym)
            if alias and alias != _yahoo_symbol(sym):
                rec["alias"] = alias
                df, _m = client.chart_with_meta(alias, period=period, interval="1d")
                if not df.empty:
                    used = "yahoo:alias"
        if df is None or df.empty:
            blocked = client.throttled_recently()
            rec.update(status=("throttled" if blocked else "no_data"),
                       issues=(("HTTP 429 (rate limited) | transport=" + client.transport)
                               if blocked else
                               "no bars from Yahoo (delisted, renamed, or newly listed)"))
            rec["elapsed"] = time.time() - t0
            return rec
        clean, issues, m = assess_daily(df, min_weekly_bars=cfg.min_weekly_bars,
                                       max_fresh_lag_days=cfg.max_fresh_lag_days)
        rec["rows"] = int(len(clean))
        rec["first"] = m.get("first_day", "") or ""
        rec["last"] = m.get("last_day", "") or ""
        rec["max_1d_move_pct"] = m.get("max_1d_move")
        rec["gap_days"] = m.get("gap_days")
        rec["ohlc_repaired"] = m.get("ohlc_repaired", 0)
        hard = [i for i in issues if i.startswith(("no_data", "bad_index", "only ~", "very stale",
                                                   "non-positive", "low liquidity"))]
        if hard:
            rec.update(status=("insufficient_history" if any(i.startswith("only ~") for i in hard)
                               else ("illiquid_history"
                                     if any(i.startswith("low liquidity") for i in hard)
                                     else "stale_data")),
                       issues="; ".join(issues))
            rec["elapsed"] = time.time() - t0
            return rec
        rec["issues"] = "; ".join(issues)
        rec["source"] = used
        # cache the raw bars even when the quality gate rejects them: the gate can be
        # relaxed later (params change) and re-downloading 1000 files is what gets you blocked
        if used.startswith("yahoo"):
            try:
                save_daily_cache(clean if not hard else df, cfg.cache_dir, sym, cfg.cache_format)
            except Exception:
                pass
        rec["elapsed"] = time.time() - t0
        with lock:
            data[sym] = clean
            report_rows[sym] = rec
        return rec

    workers = max(1, int(cfg.max_workers))
    bar = _tqdm(total=len(todo), desc="Fetching bars", unit="stock", ncols=100) \
        if (progress and _tqdm is not None) else None
    done_ct = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = {}
        nxt = 0

        def submit_next():
            nonlocal nxt
            if nxt < len(todo):
                s_ = todo[nxt]
                nxt += 1
                inflight[ex.submit(_one, s_)] = s_

        for _ in range(min(workers * 3, len(todo))):
            submit_next()
        while inflight:
            finished, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in finished:
                sym = inflight.pop(fut)
                try:
                    rec = fut.result()
                except Exception as exc:
                    rec = {"Symbol": sym, "status": "exception", "rows": 0, "source": "-",
                           "issues": f"{type(exc).__name__}: {exc}", "elapsed": 0.0,
                           "first": "", "last": "", "alias": ""}
                    report_rows[sym] = rec
                report_rows.setdefault(sym, rec)
                done_ct += 1
                if bar is not None:
                    bar.update(1)
                if on_progress is not None and (done_ct % 25 == 0 or done_ct == len(todo)):
                    on_progress({"done": done_ct, "total": len(todo), "ok": len(data),
                                 "failed": done_ct - len(data)})
                submit_next()
    if bar is not None:
        bar.close()

    # optional: true market caps from the batch-quote endpoint (best effort only)
    quotes = None
    if cfg.use_quotes and todo:
        try:
            quotes = client.quote_batch(todo)
        except Exception as exc:
            client.log(f"  market-cap enrichment skipped ({type(exc).__name__})")
    if quotes is not None and not quotes.empty:
        quotes.to_csv(os.path.join(cfg.cache_dir, "quotes_optional.csv"), index=False)

    # ---- deferred pass: symbols lost to rate limiting, retried slowly on another transport
    if cfg.final_pass:
        retry = [s for s, r in report_rows.items()
                 if r.get("rows", 0) == 0 and "429" in str(r.get("issues", ""))
                 or (r.get("status") in ("exception", "not_run"))]
        if retry:
            client.log(f"· {len(retry)} symbol(s) were throttled; slow second pass in "
                       f"{max(5, int(cfg.throttle_pause))}s…")
            client.state.schedule(cfg.throttle_pause)
            order = {"curl_cffi": "requests", "requests": "curl_cffi", "yfinance": "requests"}
            client.switch_transport(order.get(client.transport, "yfinance"))
            saved_sleep, saved_workers = cfg.request_sleep, cfg.max_workers
            cfg.request_sleep = max(0.35, saved_sleep * 3)
            cfg.max_workers = 1
            client.state = BackoffState(base_interval=cfg.request_sleep)
            for sym in retry:
                rec = _one(sym)
                report_rows[sym] = rec
            cfg.request_sleep, cfg.max_workers = saved_sleep, saved_workers
            client.log(f"  → second pass recovered "
                       f"{sum(1 for s in retry if report_rows[s].get('rows', 0) > 0)}/{len(retry)}")

    rep = pd.DataFrame([report_rows.get(s, {"Symbol": s, "status": "not_run", "rows": 0,
                                           "source": "-", "issues": "", "elapsed": 0.0,
                                           "first": "", "last": "", "alias": ""}) for s in todo])
    if not rep.empty:
        rep["status"] = rep["status"].fillna("unknown")
    return data, rep


# --------------------------------------------------------------------------------------
# Symbol alias resolution for "renamed/delisted" misses
# --------------------------------------------------------------------------------------
def resolve_missing(missing: Sequence[str], client: YahooClient,
                    universe: Optional[pd.DataFrame] = None, max_tries: int = 40) -> Dict[str, str]:
    """
    For symbols Yahoo could not serve, ask the lookup endpoint and keep any *.NS match
    whose leading token is a prefix of the original symbol (e.g. TATAMOTORS -> TATAMOTOLERS).
    Returns {old_symbol: new_yahoo_symbol} and logs the mapping.
    """
    out: Dict[str, str] = {}
    if getattr(client, "transport", "") == "yfinance":
        client.log("  (alias lookup skipped: lookup endpoint is not proxied by yfinance)")
        return out
    for sym in list(missing)[:max_tries]:
        if client.throttled_recently():
            client.log("  (alias lookup aborted: Yahoo is throttling this IP right now)")
            break
        try:
            lk = client.lookup(sym, limit=8)
        except Exception:
            continue
        if lk is None or lk.empty:
            continue
        base = sym.upper()
        for cand in lk["yahoo_symbol"].dropna().astype(str):
            if not cand.endswith(".NS"):
                continue
            head = cand[:-3].upper()
            if head == base:
                continue
            # same first 5 chars => likely a rename of the same company
            if head[:5] == base[:5] or base[:5] in head:
                out[sym] = cand
                break
    return out


# --------------------------------------------------------------------------------------
# Health report
# --------------------------------------------------------------------------------------
def data_health(report: pd.DataFrame, cfg: FetchConfig) -> str:
    lines: List[str] = []
    if report is None or report.empty:
        return "No data fetched."
    # "not_run" rows are placeholders for symbols whose worker never finished (or that were
    # supplied twice) - count them separately so 100% coverage is provable, not implied
    if "status" in report.columns:
        n = int(report.loc[report["status"] != "not_run", "status"].size)
    else:
        n = len(report)
    notrun = int((report["status"] == "not_run").sum()) if "status" in report.columns else 0
    ok = int((report["status"] == "ok").sum())
    lines.append(f"Symbols attempted : {n}" + (f"  (+{notrun} placeholder rows)" if notrun else ""))
    lines.append(f"Usable            : {ok}  ({ok/n:.1%})")
    for st, c in report["status"].value_counts().items():
        lines.append(f"   - {st}: {c}")
    thr = int((report["status"] == "throttled").sum())
    if thr:
        lines.append(f"   ! {thr} symbol(s) were rate-limited (429), NOT unavailable. "
                     f"Raise cfg.throttle_pause / lower cfg.max_workers and re-run; "
                     f"already-cached symbols are reused, so the retry is cheap.")
    if "source" in report.columns:
        lines.append("Sources: " + ", ".join(f"{k}={v}" for k, v in report["source"].value_counts().items()))
    rows = pd.to_numeric(report.get("rows", pd.Series(dtype=float)), errors="coerce")
    if rows.notna().any():
        lines.append(f"Rows/stock: median {rows.median():.0f}, min {rows.min():.0f}, max {rows.max():.0f}")
    stale = pd.to_numeric(report.get("issues", pd.Series(dtype=object)).astype(str).str.contains("stale"), errors="coerce")
    if stale.notna().any():
        lines.append(f"Staleness warnings: {int(stale.sum())}")
    lines.append(f"Minimum weekly bars required: {cfg.min_weekly_bars} (~{cfg.min_weekly_bars*5} trading days)")
    return "\n".join(lines)


def save_load_report(report: pd.DataFrame, data: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    report.to_csv(os.path.join(out_dir, "data_fetch_report.csv"), index=False)
    summary = {s: int(len(d)) for s, d in data.items()}
    pd.DataFrame({"Symbol": list(summary), "daily_rows": list(summary.values())}).to_csv(
        os.path.join(out_dir, "data_rowcounts.csv"), index=False)