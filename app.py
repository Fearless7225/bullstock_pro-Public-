import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO
from rapidfuzz import process, fuzz

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

def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "", (t or "").upper())

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

# ---------------- S&P 500 universe & resolver ----------------
@st.cache_data
def load_sp500() -> pd.DataFrame:
    # 1) Local CSV if provided: data/sp500.csv with columns Ticker,Company
    try:
        df = pd.read_csv("data/sp500.csv").dropna()
        df["Ticker"]  = df["Ticker"].str.upper().str.strip()
        df["Company"] = df["Company"].str.strip()
        if not df.empty: return df[["Ticker","Company"]]
    except Exception:
        pass
    # 2) Wikipedia (most reliable & complete for names)
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        # look for standard table
        for t in tables:
            cols = {c.lower(): c for c in t.columns}
            if "symbol" in cols and ("security" in cols or "company" in cols):
                symcol = cols["symbol"]; namecol = cols.get("security", cols.get("company"))
                w = t[[symcol, namecol]].rename(columns={symcol:"Ticker", namecol:"Company"})
                w["Ticker"]  = w["Ticker"].str.upper().str.strip()
                w["Company"] = w["Company"].str.strip()
                return w
    except Exception:
        pass
    # 3) yfinance fallback (tickers only; names = ticker)
    try:
        tickers = yf.tickers_sp500()
        if tickers:
            return pd.DataFrame({"Ticker":[t.upper() for t in tickers],
                                 "Company":[t.upper() for t in tickers]})
    except Exception:
        pass
    # 4) minimal fallback
    fallback = [
        ("AAPL","Apple Inc."), ("MSFT","Microsoft Corporation"),
        ("NVDA","NVIDIA Corporation"), ("AMZN","Amazon.com, Inc."),
        ("META","Meta Platforms, Inc."), ("GOOGL","Alphabet Inc. (Class A)"),
        ("UNH","UnitedHealth Group Incorporated"), ("ANET","Arista Networks, Inc."),
        ("ARM","Arm Holdings plc"), ("NFLX","Netflix, Inc."),
        ("PLTR","Palantir Technologies Inc.")
    ]
    return pd.DataFrame(fallback, columns=["Ticker","Company"])

SP500 = load_sp500()
_ALL_TICKERS = set(SP500["Ticker"])
_NAME_TO_TICKER = {row.Company.upper(): row.Ticker for row in SP500.itertuples()}

def name_to_ticker(user_text: str) -> str | None:
    """Intelligent resolver: name or ticker -> S&P ticker."""
    if not user_text: return None
    raw = user_text.upper().strip()
    if raw in _ALL_TICKERS:
        return raw
    choices = list(_NAME_TO_TICKER.keys())
    if choices:
        match, score, _ = process.extractOne(raw, choices, scorer=fuzz.WRatio)
        if score >= 85:
            return _NAME_TO_TICKER.get(match)
    cleaned = clean_ticker(raw)
    return cleaned if cleaned else None

def expand_input_to_tickers(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    toks = re.split(r"[,\n;| ]+", text.strip())
    out = []
    for t in toks:
        tk = name_to_ticker(t)
        if tk: out.append(tk)
    # unique, keep order
    seen=set(); dedup=[]
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

# ---------------- Scoring (0–10; missing -> 0) ----------------
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
    if p is None or pd.isna(p): return 0
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
    # Adam Khoo–style emphasis: Barriers & Scale heavier
    w = [1.0, 1.25, 1.0, 0.75, 1.25]
    v = [brand, barriers, switching, network, scale]
    return round(sum(vi*wi for vi,wi in zip(v,w)) / sum(w), 2)

def moat_note(name, label, score):
    # longer narrative
    if score >= 9:
        nuance = ("Dominant scale and regulatory/contract entrenchment; data and distribution flywheels "
                  "support pricing and returns likely to persist a decade+.")
    elif score >= 7:
        nuance = ("Clear, defendable advantages (cost curve, data, compliance, distribution). "
                  "Well‑funded rivals can narrow the gap but replication is hard.")
    elif score >= 5:
        nuance = ("Advantages exist but face credible substitutes; tech/policy shifts can compress margins; "
                  "excess returns depend on execution.")
    elif score >= 3:
        nuance = ("Some differentiation but entry frictions are modest; customer/supplier power limits pricing.")
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

def fetch_metrics(tk: str):
    t = yf.Ticker(tk)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or tk

    price = safe_price(t)
    peg = info.get("pegRatio")

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
            if de > 10:
                de = de/100.0
    except: de = None
    if de is None:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = first_val(bs.iloc[:,0], ["Total Debt","Short Long Term Debt","Short/Long Term Debt","Total Liab"])
                equity     = first_val(bs.iloc[:,0], ["Total Stockholder Equity","Total Shareholder Equity","Stockholders Equity"])
                if total_debt is not None and equity not in (None, 0):
                    de = float(total_debt)/float(equity)
        except: pass

    # FCF yield
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

    return {
        "Ticker": tk, "Company": name, "Price": price,
        "Revenue Growth YoY (%)": yoy, "Debt/Equity": de,
        "PEG Ratio": peg, "FCF ($)": fcf, "Market Cap ($)": mktcap,
        "FCF Yield (%)": fcfy,
        "Profile": f"https://finance.yahoo.com/quote/{tk}/profile"
    }

# ---------------- UI ----------------
st.sidebar.header("Economic Moat (subscores 0–10)")
brand   = st.sidebar.slider("1) Brand & Pricing", 0, 10, 6)
barrier = st.sidebar.slider("2) Barriers to Entry", 0, 10, 7)
switch  = st.sidebar.slider("3) Switching Costs", 0, 10, 6)
network = st.sidebar.slider("4) Network Effect", 0, 10, 6)
scale   = st.sidebar.slider("5) Economies of Scale", 0, 10, 8)
moat_s  = moat_avg(brand, barrier, switch, network, scale)

st.sidebar.header("Filters")
flt_moat  = st.sidebar.slider("Moat score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_total = st.sidebar.slider("Total score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_peg   = st.sidebar.slider("Max PEG", 0.0, 5.0, 5.0, 0.1)
flt_rev   = st.sidebar.slider("Min Rev Growth %", -50.0, 50.0, -50.0, 1.0)
flt_de    = st.sidebar.slider("Max Debt/Equity", 0.0, 3.0, 3.0, 0.1)
flt_fcfy  = st.sidebar.slider("Min FCF Yield %", -5.0, 15.0, -5.0, 0.5)
flt_price = st.sidebar.slider("Min Price", 0.0, 1500.0, 0.0, 1.0)

st.sidebar.header("Screener mode (S&P 500)")
use_sp500 = st.sidebar.checkbox("Scan entire S&P 500", value=False)
min_moat  = st.sidebar.slider("Screener: Min Moat", 0.0, 10.0, 8.0, 0.5)
min_rev   = st.sidebar.number_input("Screener: Min Revenue Growth %", value=23.0, step=1.0)
max_peg_s = st.sidebar.number_input("Screener: Max PEG", value=5.0, step=0.1)
max_de_s  = st.sidebar.number_input("Screener: Max Debt/Equity", value=3.0, step=0.1)
min_fcfy_s= st.sidebar.number_input("Screener: Min FCF Yield %", value=-5.0, step=0.5)

defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX, PLTR"
user_text = st.text_area("Enter tickers or company names (comma/space/newline)", defaults)

# Universe:
typed = expand_input_to_tickers(user_text or defaults)
universe = list(SP500["Ticker"]) if use_sp500 else typed
# If use_sp500 is on, also include anything the user typed explicitly (e.g., non‑S&P tickers)
if use_sp500:
    universe = sorted(set(universe).union(set(typed)))

run = st.button("Run")

if run:
    if not universe:
        st.warning("No valid tickers found. Try typing company names (e.g., 'Palantir') or enable Scan entire S&P 500.")
    rows = []
    prog = st.progress(0.0, text="Fetching...")
    for i, tk in enumerate(universe):
        m = fetch_metrics(tk)

        rev_s  = s_rev(m["Revenue Growth YoY (%)"])
        de_s   = s_de(m["Debt/Equity"])
        peg_s  = s_peg(m["PEG Ratio"])
        fcfy_s = s_fcfy(m["FCF Yield (%)"])
        total  = total_score(rev_s, de_s, fcfy_s, peg_s, moat_s)

        notes = "\n".join([
            moat_note(m["Company"], "1) Brand & Pricing", brand),
            moat_note(m["Company"], "2) Barriers to Entry", barrier),
            moat_note(m["Company"], "3) Switching Costs", switch),
            moat_note(m["Company"], "4) Network Effect", network),
            moat_note(m["Company"], "5) Economies of Scale", scale),
        ])

        rows.append({
            **m,
            "Revenue Score": rev_s, "Debt Score": de_s,
            "PEG Score": peg_s, "FCF Yield Score": fcfy_s,
            "Moat Subscores": f"{brand},{barrier},{switch},{network},{scale}",
            "Moat Score": moat_s, "Moat Notes": notes,
            "Total Score": total
        })
        prog.progress((i + 1) / max(len(universe),1), text=f"Processed {i + 1}/{len(universe)}")

    df = pd.DataFrame(rows)

    # Screener pre-filter (applies only when scanning S&P)
    screener_ok = pd.Series(True, index=df.index)
    if use_sp500:
        screener_ok = (
            df["Moat Score"].fillna(0) >= float(min_moat)
        ) & (
            df["Revenue Growth YoY (%)"].fillna(-1e9) >= float(min_rev)
        ) & (
            df["PEG Ratio"].fillna(1e9) <= float(max_peg_s)
        ) & (
            df["Debt/Equity"].fillna(1e9) <= float(max_de_s)
        ) & (
            df["FCF Yield (%)"].fillna(-1e9) >= float(min_fcfy_s)
        )

    # NaN-friendly UI filters
    peg_ok   = df["PEG Ratio"].isna() | (df["PEG Ratio"] <= flt_peg)
    rev_ok   = df["Revenue Growth YoY (%)"].isna() | (df["Revenue Growth YoY (%)"] >= flt_rev)
    de_ok    = df["Debt/Equity"].isna() | (df["Debt/Equity"] <= flt_de)
    fcfy_ok  = df["FCF Yield (%)"].isna() | (df["FCF Yield (%)"] >= flt_fcfy)
    price_ok = df["Price"].isna() | (df["Price"] >= flt_price)
    moat_ok  = df["Moat Score"].fillna(0).between(flt_moat[0], flt_moat[1])
    total_ok = df["Total Score"].fillna(0).between(flt_total[0], flt_total[1])

    filt = screener_ok & moat_ok & total_ok & peg_ok & rev_ok & de_ok & fcfy_ok & price_ok
    df_f = df[filt].reset_index(drop=True)

    # Ticker hyperlink column (to Yahoo Profile)
    df_f.insert(0, "Ticker (Profile)", df_f["Profile"])
    # show link with LinkColumn
    colcfg = {
        "Ticker (Profile)": st.column_config.LinkColumn("Ticker", help="Yahoo Finance profile", display_text="→"),
        "Profile": None  # hide raw URL if you like; comment this line to show it
    }

    st.subheader("Filtered Results")
    st.dataframe(df_f.drop(columns=[c for c in ["Profile"] if c in df_f.columns]),
                 use_container_width=True, height=480, column_config=colcfg)
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
            "- PEG (0–10, lower better): <1→10, <2→6, <3→3, else 0.\n"
            "- FCF Yield (0–10, higher better): >8%→10, >5%→8, >3%→6, >1%→4, >0%→2, else 0.\n"
            "- Moat: weighted average (Brand 1.0, Barriers 1.25, Switching 1.0, Network 0.75, Scale 1.25) with longer notes.\n"
            "- Total Score: equal-weighted average of the four metrics plus moat."
        )

    with st.expander("Raw fetched rows (debug)"):
        st.dataframe(df, use_container_width=True, height=300)

else:
    st.info("Type company names (e.g., “Palantir”) or tickers, or enable 'Scan entire S&P 500', then press Run.")
