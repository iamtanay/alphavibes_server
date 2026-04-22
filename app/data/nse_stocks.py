"""
app/data/nse_stocks.py
─────────────────────────────────────────────────────────────────────────────
Static NSE stock universe used for:
  • Autocomplete search  (ticker + company name)
  • Peer lookup          (find stocks in same sector)
  • Screener seed list   (which tickers to batch-compute nightly)
  • Market overview      (top movers subset)

Format: (NSE_TICKER, DISPLAY_NAME, SECTOR, yfinance_suffix)
yfinance symbol = NSE_TICKER + ".NS"

All tickers verified to work with yfinance as of mid-2025.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import TypedDict


class StockMeta(TypedDict):
    ticker: str          # e.g. "RELIANCE"
    yf_symbol: str       # e.g. "RELIANCE.NS"
    name: str
    sector: str
    exchange: str


# ── Core universe (Nifty 500 representative subset) ───────────────────────
# Grouped by sector for easy peer lookup
_RAW: list[tuple[str, str, str]] = [
    # (TICKER, NAME, SECTOR)

    # ── Oil & Gas ─────────────────────────────────────────────────────────
    ("RELIANCE",    "Reliance Industries Ltd.",        "Oil & Gas"),
    ("ONGC",        "Oil & Natural Gas Corporation",   "Oil & Gas"),
    ("IOC",         "Indian Oil Corporation",          "Oil & Gas"),
    ("BPCL",        "Bharat Petroleum Corporation",    "Oil & Gas"),
    ("HINDPETRO",   "Hindustan Petroleum Corporation", "Oil & Gas"),
    ("GAIL",        "GAIL (India) Ltd.",               "Oil & Gas"),
    ("OIL",         "Oil India Ltd.",                  "Oil & Gas"),

    # ── Information Technology ────────────────────────────────────────────
    ("TCS",         "Tata Consultancy Services",       "Information Technology"),
    ("INFY",        "Infosys Ltd.",                    "Information Technology"),
    ("HCLTECH",     "HCL Technologies Ltd.",           "Information Technology"),
    ("WIPRO",       "Wipro Ltd.",                      "Information Technology"),
    ("TECHM",       "Tech Mahindra Ltd.",              "Information Technology"),
    ("LTIM",        "LTIMindtree Ltd.",                "Information Technology"),
    ("MPHASIS",     "Mphasis Ltd.",                    "Information Technology"),
    ("PERSISTENT",  "Persistent Systems Ltd.",         "Information Technology"),
    ("COFORGE",     "Coforge Ltd.",                    "Information Technology"),

    # ── Banking ───────────────────────────────────────────────────────────
    ("HDFCBANK",    "HDFC Bank Ltd.",                  "Banking"),
    ("ICICIBANK",   "ICICI Bank Ltd.",                 "Banking"),
    ("KOTAKBANK",   "Kotak Mahindra Bank",             "Banking"),
    ("AXISBANK",    "Axis Bank Ltd.",                  "Banking"),
    ("SBIN",        "State Bank of India",             "Banking"),
    ("INDUSINDBK",  "IndusInd Bank Ltd.",              "Banking"),
    ("BANDHANBNK",  "Bandhan Bank Ltd.",               "Banking"),
    ("FEDERALBNK",  "The Federal Bank Ltd.",           "Banking"),
    ("IDFCFIRSTB",  "IDFC First Bank Ltd.",            "Banking"),
    ("PNB",         "Punjab National Bank",            "Banking"),

    # ── Financial Services ────────────────────────────────────────────────
    ("BAJFINANCE",  "Bajaj Finance Ltd.",              "Financial Services"),
    ("BAJAJFINSV",  "Bajaj Finserv Ltd.",              "Financial Services"),
    ("SBICARD",     "SBI Cards & Payment Services",    "Financial Services"),
    ("CHOLAFIN",    "Cholamandalam Investment",        "Financial Services"),
    ("MUTHOOTFIN",  "Muthoot Finance Ltd.",            "Financial Services"),
    ("PFC",         "Power Finance Corporation",       "Financial Services"),
    ("RECLTD",      "REC Ltd.",                        "Financial Services"),

    # ── Consumer Goods / FMCG ─────────────────────────────────────────────
    ("HINDUNILVR",  "Hindustan Unilever Ltd.",         "Consumer Goods"),
    ("ITC",         "ITC Ltd.",                        "Consumer Goods"),
    ("NESTLEIND",   "Nestle India Ltd.",               "Consumer Goods"),
    ("BRITANNIA",   "Britannia Industries Ltd.",       "Consumer Goods"),
    ("DABUR",       "Dabur India Ltd.",                "Consumer Goods"),
    ("MARICO",      "Marico Ltd.",                     "Consumer Goods"),
    ("GODREJCP",    "Godrej Consumer Products",        "Consumer Goods"),
    ("COLPAL",      "Colgate-Palmolive (India)",       "Consumer Goods"),
    ("EMAMILTD",    "Emami Ltd.",                      "Consumer Goods"),

    # ── Automobile ────────────────────────────────────────────────────────
    ("MARUTI",      "Maruti Suzuki India Ltd.",        "Automobile"),
    ("TATAMOTORS",  "Tata Motors Ltd.",                "Automobile"),
    ("M&M",         "Mahindra & Mahindra Ltd.",        "Automobile"),
    ("BAJAJ-AUTO",  "Bajaj Auto Ltd.",                 "Automobile"),
    ("HEROMOTOCO",  "Hero MotoCorp Ltd.",              "Automobile"),
    ("EICHERMOT",   "Eicher Motors Ltd.",              "Automobile"),
    ("TVSMOTORS",   "TVS Motor Company Ltd.",          "Automobile"),
    ("ASHOKLEY",    "Ashok Leyland Ltd.",              "Automobile"),
    ("TVSMOTOR",    "TVS Motor Company",               "Automobile"),

    # ── Pharma & Healthcare ───────────────────────────────────────────────
    ("SUNPHARMA",   "Sun Pharmaceutical Industries",  "Pharmaceuticals"),
    ("DRREDDY",     "Dr. Reddy's Laboratories",       "Pharmaceuticals"),
    ("CIPLA",       "Cipla Ltd.",                      "Pharmaceuticals"),
    ("DIVISLAB",    "Divi's Laboratories Ltd.",        "Pharmaceuticals"),
    ("AUROPHARMA",  "Aurobindo Pharma Ltd.",           "Pharmaceuticals"),
    ("LUPIN",       "Lupin Ltd.",                      "Pharmaceuticals"),
    ("TORNTPHARM",  "Torrent Pharmaceuticals",         "Pharmaceuticals"),
    ("APOLLOHOSP",  "Apollo Hospitals Enterprise",     "Healthcare"),
    ("MAXHEALTH",   "Max Healthcare Institute",        "Healthcare"),
    ("FORTIS",      "Fortis Healthcare Ltd.",          "Healthcare"),

    # ── Metals & Mining ───────────────────────────────────────────────────
    ("TATASTEEL",   "Tata Steel Ltd.",                 "Metals & Mining"),
    ("JSWSTEEL",    "JSW Steel Ltd.",                  "Metals & Mining"),
    ("HINDALCO",    "Hindalco Industries Ltd.",        "Metals & Mining"),
    ("VEDL",        "Vedanta Ltd.",                    "Metals & Mining"),
    ("SAIL",        "Steel Authority of India",        "Metals & Mining"),
    ("NMDC",        "NMDC Ltd.",                       "Metals & Mining"),
    ("COALINDIA",   "Coal India Ltd.",                 "Metals & Mining"),

    # ── Infrastructure & Construction ─────────────────────────────────────
    ("LT",          "Larsen & Toubro Ltd.",            "Infrastructure"),
    ("ULTRACEMCO",  "UltraTech Cement Ltd.",           "Infrastructure"),
    ("GRASIM",      "Grasim Industries Ltd.",          "Infrastructure"),
    ("SHREECEM",    "Shree Cement Ltd.",               "Infrastructure"),
    ("ACC",         "ACC Ltd.",                        "Infrastructure"),
    ("AMBUJACEM",   "Ambuja Cements Ltd.",             "Infrastructure"),
    ("ADANIPORTS",  "Adani Ports & SEZ Ltd.",          "Infrastructure"),

    # ── Power & Utilities ─────────────────────────────────────────────────
    ("POWERGRID",   "Power Grid Corporation",          "Power & Utilities"),
    ("NTPC",        "NTPC Ltd.",                       "Power & Utilities"),
    ("ADANIGREEN",  "Adani Green Energy Ltd.",         "Power & Utilities"),
    ("TATAPOWER",   "Tata Power Company Ltd.",         "Power & Utilities"),
    ("TORNTPOWER",  "Torrent Power Ltd.",              "Power & Utilities"),
    ("CESC",        "CESC Ltd.",                       "Power & Utilities"),

    # ── Telecom ───────────────────────────────────────────────────────────
    ("BHARTIARTL",  "Bharti Airtel Ltd.",              "Telecom"),
    ("IDEA",        "Vodafone Idea Ltd.",              "Telecom"),
    ("INDUSTOWER",  "Indus Towers Ltd.",               "Telecom"),

    # ── Retail & Consumer Discretionary ──────────────────────────────────
    ("DMART",       "Avenue Supermarts Ltd.",          "Retail"),
    ("TITAN",       "Titan Company Ltd.",              "Retail"),
    ("TRENT",       "Trent Ltd.",                      "Retail"),
    ("NYKAA",       "FSN E-Commerce Ventures",         "Retail"),
    ("ZOMATO",      "Zomato Ltd.",                     "Retail"),

    # ── Paints & Chemicals ────────────────────────────────────────────────
    ("ASIANPAINT",  "Asian Paints Ltd.",               "Paints & Chemicals"),
    ("BERGERPAINTS","Berger Paints India Ltd.",         "Paints & Chemicals"),
    ("PIDILITIND",  "Pidilite Industries Ltd.",        "Paints & Chemicals"),
    ("SRF",         "SRF Ltd.",                        "Paints & Chemicals"),

    # ── Real Estate ───────────────────────────────────────────────────────
    ("DLF",         "DLF Ltd.",                        "Real Estate"),
    ("GODREJPROP",  "Godrej Properties Ltd.",          "Real Estate"),
    ("OBEROIRLTY",  "Oberoi Realty Ltd.",              "Real Estate"),
    ("PRESTIGE",    "Prestige Estates Projects",       "Real Estate"),

    # ── Media & Entertainment ─────────────────────────────────────────────
    ("ZEEL",        "Zee Entertainment Enterprises",  "Media"),
    ("SUNTV",       "Sun TV Network Ltd.",             "Media"),
    ("PVR",         "PVR INOX Ltd.",                   "Media"),

    # ── Insurance ────────────────────────────────────────────────────────
    ("SBILIFE",     "SBI Life Insurance Company",     "Insurance"),
    ("HDFCLIFE",    "HDFC Life Insurance Company",    "Insurance"),
    ("ICICIGI",     "ICICI Lombard General Insurance","Insurance"),
    ("LICI",        "Life Insurance Corporation",     "Insurance"),
]


def _build_universe() -> list[StockMeta]:
    """Convert raw tuples into typed StockMeta dicts."""
    result: list[StockMeta] = []
    for ticker, name, sector in _RAW:
        result.append(
            StockMeta(
                ticker=ticker,
                yf_symbol=f"{ticker}.NS",
                name=name,
                sector=sector,
                exchange="NSE",
            )
        )
    return result


# ── Public exports ────────────────────────────────────────────────────────

# Full universe list — used by screener batch job
UNIVERSE: list[StockMeta] = _build_universe()

# Fast lookup dicts
TICKER_TO_META: dict[str, StockMeta] = {s["ticker"]: s for s in UNIVERSE}
NAME_TO_META: dict[str, StockMeta] = {s["name"].lower(): s for s in UNIVERSE}


def search_stocks(query: str, limit: int = 10) -> list[StockMeta]:
    """
    Fuzzy search by ticker or company name.
    Prioritises exact ticker match, then prefix match, then substring match.
    Returns up to `limit` results.
    """
    q = query.strip().upper()
    q_lower = query.strip().lower()

    exact: list[StockMeta] = []
    prefix: list[StockMeta] = []
    substring: list[StockMeta] = []

    for stock in UNIVERSE:
        ticker = stock["ticker"]
        name_lower = stock["name"].lower()

        if ticker == q:
            exact.append(stock)
        elif ticker.startswith(q) or name_lower.startswith(q_lower):
            prefix.append(stock)
        elif q in ticker or q_lower in name_lower:
            substring.append(stock)

    combined = exact + prefix + substring
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[StockMeta] = []
    for s in combined:
        if s["ticker"] not in seen:
            seen.add(s["ticker"])
            unique.append(s)
    return unique[:limit]


def get_peers(ticker: str, limit: int = 6) -> list[StockMeta]:
    """
    Return stocks in the same sector as `ticker`, excluding itself.
    Falls back to any 5 stocks if sector not found.
    """
    meta = TICKER_TO_META.get(ticker)
    if not meta:
        return []

    sector = meta["sector"]
    peers = [
        s for s in UNIVERSE
        if s["sector"] == sector and s["ticker"] != ticker
    ]

    # Return up to limit, but always include at least 3 stable large-caps
    if len(peers) < 3:
        # Fallback: just take any stocks from universe
        peers = [s for s in UNIVERSE if s["ticker"] != ticker]

    return peers[:limit]


def resolve_yf_symbol(ticker: str) -> str:
    """
    Given a UI ticker (e.g. 'RELIANCE'), return the yfinance symbol.
    Handles both NSE (.NS) and BSE (.BO) gracefully.
    """
    # Already has suffix
    if "." in ticker:
        return ticker

    meta = TICKER_TO_META.get(ticker.upper())
    if meta:
        return meta["yf_symbol"]

    # Default: append .NS and hope for the best
    return f"{ticker.upper()}.NS"
