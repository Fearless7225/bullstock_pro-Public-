import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO

# ---------------- Page ----------------
st.set_page_config(page_title="BullStock", layout="wide")
st.title("BullStock — Moat + Financial Screener (S&P 500)")

# ---------------- Helpers ----------------
def excel_bytes(df: pd.DataFrame, sheet="Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf.getvalue()

def clean_tok(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "", (s or "").upper())

def safe_price(tkr: yf.Ticker):
    info = tkr.info or {}
    p = info.get("currentPrice")
    if p is not None:
        try:
            return float(p)
        except Exception:
            pass
    try:
        h = tkr.history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

# -------- PEG fallbacks (so PEG isn't always missing) --------
def _eps_cagr_from_financials(tkr: yf.Ticker, info: dict) -> float | None:
    """Try to derive multi‑year EPS CAGR from Yahoo financials (returns decimal, e.g. 0.18)."""
    try:
        inc = tkr.financials
        if inc is None or inc.empty:
            return None

        # Try EPS row first
        eps_rows = [r for r in inc.index if "eps" in r.lower()]
        if eps_rows:
            s = inc.loc[eps_rows[0]].dropna()
            if len(s) >= 2:
                v0 = float(s.iloc[-1])  # oldest
                vN = float(s.iloc[0])   # latest
                n = len(s) - 1
                if v0 > 0 and vN > 0 and n > 0:
                    return (vN / v0) ** (1 / n) - 1

        # Fallback: Net Income / sharesOutstanding
        ni_rows = [r for r in inc.index if "net income" in r.lower()]
        so = info.get("sharesOutstanding")
        if ni_rows and so and so > 0:
            s = inc.loc[ni_rows[0]].dropna()
            if len(s) >= 2:
                v0 = float(s.iloc[-1]) / float(so)
                vN = float(s.iloc[0]) / float(so)
                n = len(s) - 1
                if v0 > 0 and vN > 0 and n > 0:
                    return (vN / v0) ** (1 / n) - 1
    except Exception:
        pass
    return None

def _fallback_peg(tkr: yf.Ticker, price: float | None, info: dict) -> float | None:
    """
    Compute PEG if Yahoo's pegRatio is missing.
    PEG = (P/E) / (EPS CAGR in %) using trailingPE or forward EPS.
    """
    pe = info.get("trailingPE")
    if pe is None and price is not None:
        fwd_eps = info.get("forwardEps")
        if fwd_eps:
            try:
                pe = float(price) / float(fwd_eps)
            except Exception:
                pe = None

    g = _eps_cagr_from_financials(tkr, info)  # decimal, e.g. 0.18 for 18%
    if pe and g and g > 0:
        try:
            return float(pe) / (float(g) * 100.0)  # divide by growth in %
        except Exception:
            return None
    return None

# ---------------- S&P 500 universe ----------------
@st.cache_data
def load_sp500() -> pd.DataFrame:
    # Try Wikipedia (tickers + names)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            cols = {c.lower(): c for c in t.columns}
            if "symbol" in cols and ("security" in cols or "company" in cols):
                symcol = cols["symbol"]
                namecol = cols.get("security", cols.get("company"))
                w = t[[symcol, namecol]].rename(columns={symcol: "Ticker", namecol: "Company"})
                w["Ticker"] = w["Ticker"].str.upper().str.strip()
                w["Company"] = w["Company"].str.strip()
                return w[["Ticker", "Company"]]
    except Exception:
        pass
    # Minimal fallback so app still runs
    fallback = [
        ("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corporation"),
        ("NVDA", "NVIDIA Corporation"), ("AMZN", "Amazon.com, Inc."),
        ("META", "Meta Platforms, Inc."), ("GOOGL", "Alphabet Inc. (Class A)"),
        ("UNH", "UnitedHealth Group Incorporated"), ("ANET", "Arista Networks, Inc."),
        ("ARM", "Arm Holdings plc"), ("NFLX", "Netflix, Inc."), ("PLTR", "Palantir Technologies Inc.")
    ]
    return pd.DataFrame(fallback, columns=["Ticker", "Company"])

SP500 = load_sp500()
_ALL_TICKERS = set(SP500["Ticker"])

# ---------------- Name / ticker resolver ----------------
def name_to_ticker(user_text: str) -> str | None:
    """Resolve company name OR raw ticker. Always accept ticker-like tokens."""
    if not user_text:
        return None
    raw = user_text.strip()
    cleaned = clean_tok(raw)

    # Already an S&P ticker?
    if cleaned in _ALL_TICKERS:
        return cleaned

    # Looks like a ticker (1–6 chars A/Z/0-9/.-)? accept anyway
    if 1 <= len(cleaned) <= 6 and re.fullmatch(r"[A-Z0-9.\-]{1,6}", cleaned):
        return cleaned

    # Otherwise, company name contains match
    cand = SP500[SP500["Company"].str.upper().str.contains(re.escape(raw.upper()))]
    if not cand.empty:
        return cand.iloc[0]["Ticker"]

    # Last resort
    return cleaned or None

def expand_input_to_tickers(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    toks = re.split(r"[,\n;| ]+", text.strip())
    out = []
    for t in toks:
        tk = name_to_ticker(t)
        if tk:
            out.append(tk)
    # unique, keep order
    seen = set(); dedup = []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

# ---------------- Scoring (0–10; missing -> 0, except PEG neutral=5) ----------------
def s_rev(y):
    if y is None or pd.isna(y): return 0
    if y > 20: return 10
    if y > 10: return 8
    if y > 5:  return 6
    if y > 0:  return 4
    return 0

def s_de(x):
    if x is None or pd.isna(x): return 0
    if x < 0.5: return 10
    if x < 1.0: return 7
    if x < 2.0: return 4
    return 0

def s_peg(p):
    # PEG missing -> neutral 5 (not punitive)
    if p is None or pd.isna(p): return 5
    if p < 1:  return 10
    if p < 2:  return 6
    if p < 3:  return 3
    return 0

def s_fcfy(y):
    if y is None or pd.isna(y): return 0
    if y > 8:  return 10
    if y > 5:  return 8
    if y > 3:  return 6
    if y > 1:  return 4
    if y > 0:  return 2
    return 0

def moat_avg(brand, barriers, switching, network, scale):
    # Adam Khoo–style: Barriers & Scale weighted higher
    w = [1.0, 1.25, 1.0, 0.75, 1.25]
    v = [brand, barriers, switching, network, scale]
    return round(sum(vi*wi for vi, wi in zip(v, w)) / sum(w), 2)

def moat_note(name, label, score):
    if score >= 9:
        nuance = ("Dominant scale and regulatory/contract entrenchment; data & distribution flywheels "
                  "support pricing power and decade‑long returns.")
    elif score >= 7:
        nuance = ("Clear, defendable advantages (cost curve, data, compliance, distribution). "
                  "Replication is hard though not impossible for well‑funded rivals.")
    elif score >= 5:
        nuance = ("Advantages exist but face credible substitutes; tech/policy shifts may compress margins; "
                  "excess returns depend on execution.")
    elif score >= 3:
        nuance = ("Some differentiation but modest entry frictions; customer/supplier power limits pricing.")
    else:
        nuance = ("Commodity dynamics with little pricing power; durable excess returns unlikely.")
    return f"{label}: {score}/10 — {nuance} ({name})."

def total_score(rev_s, de_s, fcfy_s, peg_s, moat_s):
    return round((rev_s + de_s + fcfy_s + peg_s + moat_s) / 5.0, 2)

# ---------------- Data fetch ----------------
def first_val(series, keys):
    for k in keys:
        if k in series and pd.notna(series[k]):
            return series[k]
    return None

def fetch_snapshot(tk: str):
    t = yf.Ticker(tk)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or tk

    price = safe_price(t)

    # PEG with fallback
    peg = info.get("pegRatio")
    if peg is None or pd.isna(peg):
        peg = _fallback_peg(t, price, info)

    # Revenue growth YoY (%)
    yoy = None
    if info.get("revenueGrowth") is not None:
        try:
            rg = float(info["revenueGrowth"])
            yoy = rg * 100 if abs(rg) <= 1 else rg
        except Exception:
            pass
    if yoy is None:
        try:
            inc = t.financials
            if inc is not None and not inc.empty:
                k = [i for i in inc.index if "total" in i.lower() and "revenue" in i.lower()]
                if k:
                    rev = inc.loc[k[0]].dropna()
                    if len(rev) >= 2 and float(rev.iloc[1]) != 0:
                        yoy = float((rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100.0)
        except Exception:
            pass

    # Debt/Equity
    de = info.get("debtToEquity")
    try:
        if de is not None:
            de = float(de)
            if de > 10:  # sometimes percent-like
                de = de / 100.0
    except Exception:
        de = None
    if de is None:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = first_val(bs.iloc[:, 0], ["Total Debt", "Short Long Term Debt", "Short/Long Term Debt", "Total Liab"])
                equity = first_val(bs.iloc[:, 0], ["Total Stockholder Equity", "Total Shareholder Equity", "Stockholders Equity"])
                if total_debt is not None and equity not in (None, 0):
                    de = float(total_debt) / float(equity)
        except Exception:
            pass

    # FCF & FCF Yield
    fcf = info.get("freeCashflow")
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf = first_val(cf.iloc[:, 0], ["Total Cash From Operating Activities", "Operating Cash Flow"])
                capx = first_val(cf.iloc[:, 0], ["Capital Expenditures", "Capital Expenditure"])
                if ocf is not None and capx is not None:
                    fcf = float(ocf) + float(capx)  # capex usually negative
        except Exception:
            pass
    mktcap = info.get("marketCap")
    fcfy = (float(fcf) / float(mktcap) * 100.0) if (fcf and mktcap and mktcap > 0) else None

    # scores
    rev_s  = s_rev(yoy)
    de_s   = s_de(de)
    peg_s  = s_peg(peg)
    fcfy_s = s_fcfy(fcfy)

    # moat (uses sidebar selections)
    company_moat = moat_avg(brand, barriers, switching, network, scale)
    notes = "\n".join([
        moat_note(name, "1) Brand & Pricing", brand),
        moat_note(name, "2) Barriers to Entry", barriers),
        moat_note(name, "3) Switching Costs", switching),
        moat_note(name, "4) Network Effect", network),
        moat_note(name, "5) Economies of Scale", scale),
    ])

    total = total_score(rev_s, de_s, fcfy_s, peg_s, company_moat)

    return {
        "Ticker": tk, "Company": name, "Price": price,
        "Revenue Growth YoY (%)": yoy, "Debt/Equity": de, "PEG Ratio": peg,
        "FCF ($)": fcf, "Market Cap ($)": mktcap, "FCF Yield (%)": fcfy,
        "Revenue Score": rev_s, "Debt Score": de_s, "PEG Score": peg_s, "FCF Yield Score": fcfy_s,
        "Moat Subscores": f"{brand},{barriers},{switching},{network},{scale}",
        "Moat Score": company_moat, "Moat Notes": notes,
        "Total Score": total,
        "Profile": f"https://finance.yahoo.com/quote/{tk}/profile"
    }

# ---------------- Sidebar controls ----------------
st.sidebar.header("Economic Moat (subscores 0–10)")
brand     = st.sidebar.slider("1) Brand & Pricing",      0, 10, 6)
barriers  = st.sidebar.slider("2) Barriers to Entry",    0, 10, 7)
switching = st.sidebar.slider("3) Switching Costs",      0, 10, 6)
network   = st.sidebar.slider("4) Network Effect",       0, 10, 6)
scale     = st.sidebar.slider("5) Economies of Scale",   0, 10, 8)

st.sidebar.header("Filters")
flt_moat  = st.sidebar.slider("Moat score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_total = st.sidebar.slider("Total score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_rev   = st.sidebar.slider("Min Revenue Growth %", -50.0, 50.0, -50.0, 1.0)
flt_peg   = st.sidebar.slider("Max PEG", 0.0, 10.0, 10.0, 0.1)
flt_de    = st.sidebar.slider("Max Debt/Equity", 0.0, 5.0, 5.0, 0.1)
flt_fcfy  = st.sidebar.slider("Min FCF Yield %", -10.0, 20.0, -10.0, 0.5)
flt_price = st.sidebar.slider("Min Price ($)", 0.0, 2000.0, 0.0, 1.0)

st.sidebar.header("Screener mode (S&P 500)")
scan_all = st.sidebar.checkbox("Scan entire S&P 500", value=False)
min_moat = st.sidebar.slider("Screener: Min Moat", 0.0, 10.0, 8.0, 0.5)
min_rev  = st.sidebar.number_input("Screener: Min Revenue Growth %", value=23.0, step=1.0)
max_peg  = st.sidebar.number_input("Screener: Max PEG", value=5.0, step=0.1)
max_de   = st.sidebar.number_input("Screener: Max Debt/Equity", value=3.0, step=0.1)
min_fcfy = st.sidebar.number_input("Screener: Min FCF Yield %", value=-5.0, step=0.5)
defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX, PLTR"
user_text = st.text_area("Enter tickers or company names (comma/space/newline)", defaults)
else:
    st.info("Type names or tickers (e.g., ‘Palantir’ or ‘PLTR’) or enable ‘Scan entire S&P 500’, then press Run.")
