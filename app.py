import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO

# ================== Page ==================
st.set_page_config(page_title="BullStock", layout="wide")
st.title("BullStock — Moat + Financial Screener (S&P 500)")

# ================== Helpers ==================
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
        try: return float(p)
        except: pass
    try:
        h = tkr.history(period="1d")
        if not h.empty: return float(h["Close"].iloc[-1])
    except: pass
    return None

# ---- PEG fallbacks (so PEG isn’t often missing) ----
def _eps_cagr_from_financials(tkr: yf.Ticker, info: dict) -> float | None:
    """Return EPS CAGR as decimal (0.18 for 18%) if derivable, else None."""
    try:
        inc = tkr.financials
        if inc is None or inc.empty: return None

        eps_rows = [r for r in inc.index if "eps" in r.lower()]
        if eps_rows:
            s = inc.loc[eps_rows[0]].dropna()
            if len(s) >= 2:
                v0 = float(s.iloc[-1]); vN = float(s.iloc[0]); n = len(s) - 1
                if v0 > 0 and vN > 0 and n > 0:
                    return (vN / v0) ** (1 / n) - 1

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
    except:  # noqa: E722
        pass
    return None

def _fallback_peg(tkr: yf.Ticker, price: float | None, info: dict) -> float | None:
    """
    PEG = (P/E) / (EPS CAGR in %) using trailingPE or price/forwardEps.
    """
    pe = info.get("trailingPE")
    if pe is None and price is not None:
        fwd_eps = info.get("forwardEps")
        if fwd_eps:
            try: pe = float(price) / float(fwd_eps)
            except: pe = None
    g = _eps_cagr_from_financials(tkr, info)  # decimal (0.18)
    if pe and g and g > 0:
        try: return float(pe) / (float(g) * 100.0)
        except: return None
    return None

# ================== S&P500 universe ==================
@st.cache_data
def load_sp500() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            cols = {c.lower(): c for c in t.columns}
            if "symbol" in cols and ("security" in cols or "company" in cols):
                sym = cols["symbol"]; name = cols.get("security", cols.get("company"))
                w = t[[sym, name]].rename(columns={sym:"Ticker", name:"Company"})
                w["Ticker"]  = w["Ticker"].str.upper().str.strip()
                w["Company"] = w["Company"].str.strip()
                return w[["Ticker","Company"]]
    except:  # noqa: E722
        pass
    # minimal fallback so the app runs
    fallback = [
        ("AAPL","Apple Inc."), ("MSFT","Microsoft Corporation"),
        ("NVDA","NVIDIA Corporation"), ("AMZN","Amazon.com, Inc."),
        ("META","Meta Platforms, Inc."), ("GOOGL","Alphabet Inc. (Class A)"),
        ("UNH","UnitedHealth Group Incorporated"), ("ANET","Arista Networks, Inc."),
        ("ARM","Arm Holdings plc"), ("NFLX","Netflix, Inc."), ("PLTR","Palantir Technologies Inc.")
    ]
    return pd.DataFrame(fallback, columns=["Ticker","Company"])

SP500 = load_sp500()
_ALL_TICKERS = set(SP500["Ticker"])

# ================== Resolver ==================
def name_to_ticker(user_text: str) -> str | None:
    """Resolve company name OR raw ticker. Always accept ticker-like tokens."""
    if not user_text: return None
    raw = user_text.strip()
    cleaned = clean_tok(raw)

    if cleaned in _ALL_TICKERS:  # exact ticker match in S&P
        return cleaned

    if 1 <= len(cleaned) <= 6 and re.fullmatch(r"[A-Z0-9.\-]{1,6}", cleaned):
        return cleaned  # accept typed ticker even if not in S&P

    cand = SP500[SP500["Company"].str.upper().str.contains(re.escape(raw.upper()))]
    if not cand.empty:
        return cand.iloc[0]["Ticker"]

    return cleaned or None

def expand_input_to_tickers(text: str) -> list[str]:
    if not (text or "").strip(): return []
    toks = re.split(r"[,\n;| ]+", text.strip())
    out = []
    for t in toks:
        tk = name_to_ticker(t)
        if tk: out.append(tk)
    seen=set(); dedup=[]
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

# ================== Scoring ==================
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
    if p is None or pd.isna(p): return 5  # neutral when PEG missing
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
    w = [1.0, 1.25, 1.0, 0.75, 1.25]  # Adam‑Khoo style emphasis on Barriers/Scale
    v = [brand, barriers, switching, network, scale]
    return round(sum(vi*wi for vi,wi in zip(v,w)) / sum(w), 2)

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

# ================== Data fetch ==================
def first_val(series, keys):
    for k in keys:
        if k in series and pd.notna(series[k]): return series[k]
    return None

def fetch_snapshot(tk: str):
    t = yf.Ticker(tk)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or tk

    price = safe_price(t)

    # PEG (with fallback)
    peg = info.get("pegRatio")
    if peg is None or pd.isna(peg):
        peg = _fallback_peg(t, price, info)

    # Revenue growth YoY (%)
    yoy = None
    if info.get("revenueGrowth") is not None:
        try:
            rg = float(info["revenueGrowth"])
            yoy = rg*100 if abs(rg) <= 1 else rg
        except: pass
    if yoy is None:
        try:
            inc = t.financials
            if inc is not None and not inc.empty:
                k = [i for i in inc.index if "total" in i.lower() and "revenue" in i.lower()]
                if k:
                    rev = inc.loc[k[0]].dropna()
                    if len(rev) >= 2 and float(rev.iloc[1]) != 0:
                        yoy = float((rev.iloc[0]-rev.iloc[1]) / rev.iloc[1] * 100.0)
        except: pass

    # Debt/Equity
    de = info.get("debtToEquity")
    try:
        if de is not None:
            de = float(de)
            if de > 10: de = de/100.0  # some feeds look like %
    except: de = None
    if de is None:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = first_val(bs.iloc[:,0], ["Total Debt","Short Long Term Debt","Short/Long Term Debt","Total Liab"])
                equity     = first_val(bs.iloc[:,0], ["Total Stockholder Equity","Total Shareholder Equity","Stockholders Equity"])
                if total_debt is not None and equity not in (None,0):
                    de = float(total_debt)/float(equity)
        except: pass

    # FCF & FCF Yield
    fcf = info.get("freeCashflow")
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf  = first_val(cf.iloc[:,0], ["Total Cash From Operating Activities","Operating Cash Flow"])
                capx = first_val(cf.iloc[:,0], ["Capital Expenditures","Capital Expenditure"])
                if ocf is not None and capx is not None:
                    fcf = float(ocf) + float(capx)  # capex usually negative
        except: pass
    mktcap = info.get("marketCap")
    fcfy = (float(fcf)/float(mktcap)*100.0) if (fcf and mktcap and mktcap>0) else None

    # Scores
    rev_s  = s_rev(yoy)
    de_s   = s_de(de)
    peg_s  = s_peg(peg)
    fcfy_s = s_fcfy(fcfy)

    # Moat (from sidebar sliders)
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

# ================== Sidebar controls ==================
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

# ================== Main input + Run ==================
defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX, PLTR"
user_text = st.text_area("Enter tickers or company names (comma/space/newline)", defaults)

typed = expand_input_to_tickers(user_text or defaults)
universe = list(SP500["Ticker"]) if scan_all else typed
if scan_all:
    universe = sorted(set(universe).union(set(typed)))  # include any typed extras

run = st.button("Run", type="primary")

# ================== Execute ==================
if run:
    if not universe:
        st.warning("No valid tickers found. Try typing names (e.g., 'Palantir') or enable S&P scan.")
    rows = []
    prog = st.progress(0.0, text="Fetching...")
    for i, tk in enumerate(universe):
        try:
            rows.append(fetch_snapshot(tk))
        except Exception as e:
            st.warning(f"Fetch failed for {tk}: {e}")
        prog.progress((i+1)/max(len(universe),1), text=f"Processed {i+1}/{len(universe)}")

    df = pd.DataFrame(rows)

    # Screener pre-filter (only when scanning S&P)
    screener_ok = pd.Series(True, index=df.index)
    if scan_all:
        screener_ok = (
            df["Moat Score"].fillna(0) >= float(min_moat)
        ) & (
            df["Revenue Growth YoY (%)"].fillna(-1e9) >= float(min_rev)
        ) & (
            df["PEG Ratio"].fillna(1e9) <= float(max_peg)
        ) & (
            df["Debt/Equity"].fillna(1e9) <= float(max_de)
        ) & (
            df["FCF Yield (%)"].fillna(-1e9) >= float(min_fcfy)
        )

    # UI filters (NaN-friendly)
    moat_ok  = df["Moat Score"].fillna(0).between(flt_moat[0], flt_moat[1])
    total_ok = df["Total Score"].fillna(0).between(flt_total[0], flt_total[1])
    rev_ok   = df["Revenue Growth YoY (%)"].isna() | (df["Revenue Growth YoY (%)"] >= flt_rev)
    peg_ok   = df["PEG Ratio"].isna() | (df["PEG Ratio"] <= flt_peg)
    de_ok    = df["Debt/Equity"].isna() | (df["Debt/Equity"] <= flt_de)
    fcfy_ok  = df["FCF Yield (%)"].isna() | (df["FCF Yield (%)"] >= flt_fcfy)
    price_ok = df["Price"].isna() | (df["Price"] >= flt_price)

    filt = screener_ok & moat_ok & total_ok & rev_ok & peg_ok & de_ok & fcfy_ok & price_ok
    df_f = df[filt].reset_index(drop=True)

    # Link column (ticker -> Yahoo profile)
    df_f.insert(1, "Profile Link", df_f["Profile"])
    colcfg = {
        "Profile Link": st.column_config.LinkColumn("Ticker (link)", help="Open Yahoo profile", display_text="→"),
        "Profile": None  # hide raw URL
    }

    st.subheader("Filtered Results")
    st.dataframe(
        df_f.drop(columns=[c for c in ["Profile"] if c in df_f.columns]),
        use_container_width=True, height=520, column_config=colcfg
    )
    st.caption(f"Showing {len(df_f)} of {len(df)} rows")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download FILTERED (Excel)",
            data=excel_bytes(df_f, "Filtered"),
            file_name="bullstock_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download FILTERED (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_filtered.csv", mime="text/csv")
    with c2:
        st.download_button("Download ALL (Excel)",
            data=excel_bytes(df, "All"),
            file_name="bullstock_all.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download ALL (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_all.csv", mime="text/csv")

    with st.expander("Appendix: Scoring Rules"):
        st.markdown(
            "- Revenue Growth YoY (0–10): >20%→10, >10%→8, >5%→6, >0%→4, else 0.\n"
            "- Debt/Equity (0–10, lower better): <0.5→10, <1.0→7, <2.0→4, else 0.\n"
            "- PEG (0–10, lower better): <1→10, <2→6, <3→3, else 0; **missing PEG = neutral 5**.\n"
            "- FCF Yield (0–10, higher better): >8%→10, >5%→8, >3%→6, >1%→4, >0%→2, else 0.\n"
            "- Moat: weighted avg (Brand 1.0, Barriers 1.25, Switching 1.0, Network 0.75, Scale 1.25) with narrative notes.\n"
            "- Total Score: equal‑weighted average of Revenue, D/E, FCF‑Yield, PEG, and Moat."
        )
else:
    st.info("Type names/tickers (e.g., ‘Palantir’ or ‘PLTR’) or enable ‘Scan entire S&P 500’, then press Run.")
