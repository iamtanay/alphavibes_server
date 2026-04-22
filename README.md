# AlphaVibes Server

**Stock analysis API for [AlphaVibes by Accrion](https://github.com/accrion/alphavibes)**

Production-grade FastAPI backend. Computes technical indicators, fundamental ratios, and investor persona scores entirely from free data — no paid API keys required.

---

## What this does

| Endpoint | What it returns |
|---|---|
| `GET /api/analyse/:ticker` | Full stock analysis — 30+ indicators, 50+ ratios, 9 persona scores, chart data |
| `GET /api/quote/:ticker` | Lightweight price quote — fast, uncached |
| `GET /api/search?q=` | Ticker and company name autocomplete |
| `GET /api/market/overview` | Nifty 50, Sensex, trending stocks, top movers |
| `GET /api/screener` | Pre-computed Nifty 500 screener with 15+ filters |
| `GET /api/compare?tickers=A,B,C` | Side-by-side analysis of 2–3 stocks |
| `POST /api/session/check` | Rate limit status for current session |
| `GET /health` | Health check for Railway/Render uptime monitoring |

---

## Architecture Decision: Runtime + Cache

Full analysis runs **on demand** when requested, with results cached in Redis for 15 minutes.

**Why not nightly pre-compute?**
- yfinance gets rate-limited at ~100 calls/hour. Pre-computing 500 stocks would reliably get blocked.
- With Redis caching, a second request for any stock is **instant** (sub-millisecond from cache).
- Zero cost when idle — Railway's hobby tier sleeps the server between requests.
- Any NSE/BSE stock can be analysed, not just a pre-computed list.

**Screener exception:** The screener pre-computes 15 lightweight fields (price, PE, ROE, RSI proxy) for the top 100 NSE stocks and caches them for 24 hours. This is a separate lightweight batch — not the full analysis pipeline.

---

## Quick Start (Local Development)

```bash
# 1. Clone / copy the backend folder
cd alphavibes_backend

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — for local dev, only ALLOWED_ORIGINS matters.
# Leave REDIS_URL blank — the server runs fine without cache.

# 5. Start the server
uvicorn app.main:app --reload --port 8000

# Server is now running at http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Connect the UI:**
```bash
# In your alphavibes Next.js project, create .env.local:
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > ../alphavibes-main/.env.local
```

**Test it works:**
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"alphavibes-backend"}

curl http://localhost:8000/api/quote/RELIANCE
# {"ticker":"RELIANCE","name":"Reliance Industries Ltd.",...}
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -v
# 57 passed — all services tested offline with synthetic data
```

---

## Free Data Sources

| Data | Source | Cost | Notes |
|---|---|---|---|
| Stock prices, OHLCV | yfinance (Yahoo Finance) | Free | 15-min delayed |
| Fundamentals (PE, ROE, etc.) | yfinance `info` | Free | Updated daily |
| Financial statements | yfinance financials | Free | Annual + quarterly |
| Shareholding | yfinance `heldPercentInsiders` | Partial | Promoter/FII estimate |
| Indices (Nifty, Sensex) | yfinance `^NSEI`, `^BSESN` | Free | 15-min delayed |

**yfinance limits:** Yahoo Finance has no published rate limit, but aggressive fetching gets you temporarily blocked. The caching layer ensures each stock is fetched at most once per 15 minutes.

---

## Deploy to Railway (Recommended — $5/mo)

Railway is the ideal host: persistent server, automatic deploys from GitHub, easy env vars.

```bash
# 1. Push this folder to a GitHub repo

# 2. Go to railway.app → New Project → Deploy from GitHub
#    Select your repo → Railway auto-detects the Procfile

# 3. Add environment variables in Railway dashboard:
ALLOWED_ORIGINS=https://your-app.vercel.app
REDIS_URL=rediss://...  # From Upstash (see below)
LOG_LEVEL=INFO

# 4. Railway gives you a URL like: https://alphavibes-backend.up.railway.app
#    Set this in your Vercel frontend:
NEXT_PUBLIC_API_BASE_URL=https://alphavibes-backend.up.railway.app
```

**Cost:** Railway Hobby plan = $5/month. Includes 500 hours/month (more than enough for a side project).

---

## Free Redis with Upstash

Upstash offers a free Redis tier: 10,000 commands/day — sufficient for ~600 analyses/day.

```
1. Go to https://upstash.com → Sign up free (no credit card)
2. Create Redis database → choose "Global" region for lowest latency
3. Copy the "Redis URL" (starts with rediss://)
4. Set REDIS_URL=rediss://... in your Railway environment variables
```

Without Redis, the server works fine — every request just hits yfinance directly, which takes 3–8 seconds for a full analysis instead of <100ms from cache.

---

## Deploy to Render (Alternative — Free tier available)

```bash
# render.yaml is included for zero-config Render deployment
# Go to render.com → New → Web Service → Connect GitHub repo
# Build command: pip install -r requirements.txt
# Start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Note:** Render's free tier spins down after 15 minutes of inactivity. The first request after sleep takes ~30 seconds (cold start). Use Railway's $5/mo plan to avoid this.

---

## Project Structure

```
alphavibes_backend/
├── app/
│   ├── main.py              # FastAPI app factory, CORS, middleware
│   ├── routers/
│   │   └── api.py           # All route handlers (thin HTTP layer)
│   ├── services/
│   │   ├── fetcher.py       # yfinance async wrapper (thread-pool)
│   │   ├── technicals.py    # 30+ indicators via pandas-ta
│   │   ├── fundamentals.py  # All ratios, financial statements
│   │   ├── personas.py      # 9 persona scorers — zero LLM, pure rules
│   │   ├── analyser.py      # Orchestrates all services into one response
│   │   ├── shareholding.py  # Shareholding pattern from yfinance
│   │   └── cache.py         # Redis cache (optional, graceful fallback)
│   └── data/
│       └── nse_stocks.py    # 100+ NSE stock universe for search + peers
├── tests/
│   └── test_backend.py      # 57 tests, all offline (no network)
├── requirements.txt
├── Procfile                 # Railway deployment
├── railway.toml             # Railway config
├── .env.example             # All env vars documented
└── .gitignore
```

---

## Personas (Zero LLM)

9 investor personas are scored entirely from yfinance data using explicit weighted rules:

| Persona | Key signal |
|---|---|
| Warren Buffett | ROE > 15%, ROIC > 15%, low debt, moat margins |
| Benjamin Graham | EV/EBITDA < 7x, P/B < 1.5, current ratio > 2 |
| Peter Lynch | PEG < 1.0, EPS growth > 15%, insider holding |
| Growth Investor | Revenue growth > 25%, gross margin > 40% |
| Momentum Trader | Price > SMA200, RSI 40–70, 6M momentum |
| Jim Simons | RSI zone, volume spike, MACD signal |
| Cathie Wood | Revenue growth > 40%, disruptive sector |
| Mark Minervini | Stage 2 uptrend, relative strength, VCP |
| Rakesh Jhunjhunwala | India CAGR > 20%, promoter holding, mid-cap |

Each persona returns: `score /100`, `verdict label`, `top criteria with pass/fail`, and a plain-English summary.

---

## Technical Indicators Computed

**Trend:** SMA (10/20/50/100/200), EMA (9/12/21/26/50/200), DEMA, TEMA, HMA, ZLEMA, VWMA, ALMA, KAMA, Ichimoku, MACD, Supertrend, Parabolic SAR, ADX, Aroon, Vortex, Donchian Channel

**Momentum:** RSI (with divergence detection), Stochastic, StochRSI, CCI, Williams %R, MFI, ROC, TRIX, PPO, Ultimate Oscillator, Fisher Transform, Coppock, KST, QQE, CRSI, TSI

**Volatility:** Bollinger Bands (%B, Width), ATR, Keltner Channel, Historical Volatility, Ulcer Index, Squeeze Momentum, Nadaraya-Watson Envelope

**Volume:** OBV, VWAP, A/D Line, CMF, Force Index, VROC, Ease of Movement, NVI, Klinger Oscillator

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (set by Railway automatically) |
| `LOG_LEVEL` | `INFO` | Logging level: DEBUG/INFO/WARNING/ERROR |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins (comma-separated) |
| `REDIS_URL` | _(empty)_ | Redis connection string — leave blank for no cache |
| `RATE_LIMIT` | `3` | Analyses per session window |
| `RATE_WINDOW_SECONDS` | `18000` | Rate limit window (18000 = 5 hours) |

---

## Adding a New Data Source (V2 upgrade path)

To upgrade from yfinance to a paid provider (e.g. Upstox for real-time India data):

1. Create `app/services/fetcher_upstox.py` implementing the same `RawStockData` interface
2. Change the import in `app/services/analyser.py` from `fetcher` to `fetcher_upstox`
3. All other services (technicals, fundamentals, personas) are data-source agnostic — they only receive DataFrames and dicts, and never touch yfinance directly

The entire system is designed for this swap to be a one-file change.

---

**Built with ❤️ by Accrion**
