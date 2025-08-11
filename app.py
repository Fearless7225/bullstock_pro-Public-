import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO

# ---------- helpers (downloads) ----------
def _excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()

# ---------- ticker cleaning & safe price ----------
def _clean_ticker(tok: str) -> str:
    # keep letters, numbers, dot, dash; remove $, spaces, emojis, etc.
    return re.sub(r"[^A-Za-z0-9\.\-]", "", (tok or "").upper())

def _safe_last_price(tkr: yf.Ticker):
    info = tkr.info or {}
    p = info.get("currentPrice")
    if p:
        return float(p)
    try:
        hist = tkr.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

# ---------- scoring ----------
def score_revenue_growth(yoy_pct):
    if yoy_pct is None or np.isnan(yoy_pct): return np.nan
    if yoy_pct > 20: return 10
    if yoy_pct > 10: return 8
    if yoy_pct > 5:  return 6
    if yoy_pct > 0:  return 4
    return 0

def score_debt_ratio(de_ratio):
    if de_ratio is None or np.isnan(de_ratio): return np.nan
    if de_ratio < 0.5: return 10
    if de_ratio < 1.0: return 7
    if de_ratio < 2.0: return 4
    return 0

def score_peg(peg):
    if peg is None or np.isnan(peg): return np.nan
    if peg < 1:  return 10
    if peg < 2:  return 6
    if peg < 3:  return 3
    return 0

def score_fcf_yield(fcf_yield_pct):
    if fcf_yield_pct is None or np.isnan(fcf_yield_pct): return np.nan
    if fcf_yield_pct > 8:  return 10
    if fcf_yield_pct > 5:  return 8
    if fcf_yield_pct > 3:  return 6
    if fcf_yield_pct > 1:  return 4
    if fcf_yield_pct > 0:  return 2
    return 0

def moat_composite(brand, barriers, switching, network, scale):
    subs=[brand,barriers,switching,network,scale]
    if any(s is None for s in subs): return np.nan
    return round(sum(subs)/5,2)

def total_score(rev_s, debt_s, fcfy_s, peg_s, moat_s,
                w_rev=0.2, w_debt=0.2, w_fcfy=0.2, w_peg=0.2, w_moat=0.2):
    parts=[rev_s,debt_s,fcfy_s,peg_s,moat_s]
    if any(x is None or np.isnan(x) for x in parts): return np.nan
    return round(w_rev*rev_s + w_debt*debt_s + w_fcfy*fcfy_s + w_peg*peg_s + w_moat*moat_s, 2)

def moat_justification(name, criterion, score):
    bands = [
        (9,10,"Exceptional, structurally defended; hard to replicate."),
        (7,8,"Strong but not absolute; clear edge with some constraints."),
        (5,6,"Moderate; present but contested or limited by alternatives."),
        (3,4,"Weak; easily matched by peers."),
        (0,2,"Minimal; commoditized or transient.")
    ]
    for lo, hi, text in bands:
        if score>=lo and score<=hi:
            return f"{criterion}: {score}/10 — {text} ({name})."
    return f"{criterion}: {score}/10."

# ---------- data fetch ----------
def _first(series, keys):
    for k in keys:
        if k in series and pd.notna(series[k]): return series[k]
    return np.nan

def fetch_snapshot(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or ticker
    price = _safe_last_price(t)
    peg = info.get("pegRatio", np.nan)

    # Revenue growth YoY (%)
    yoy = np.nan
    if info.get("revenueGrowth") is not None:
        try: yoy = float(info["revenueGrowth"])*100
        except: pass
    if np.isnan(yoy):
        try:
            inc = t.financials
            if inc is not None and not inc.empty:
                rev_key = [k for k in inc.index if "total" in k.lower() and "revenue" in k.lower()]
                if rev_key:
                    rev = inc.loc[rev_key[0]].dropna()
                    if len(rev)>=2:
                        yoy = float((rev.iloc[0]-rev.iloc[1]) / rev.iloc[1] * 100)
        except: pass

    # Debt/Equity
    de = info.get("debtToEquity", np.nan)
    try:
        if pd.notna(de) and de>10: de = de/100.0
    except: pass
    if np.isnan(de):
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = _first(bs.iloc[:,0], ["Total Debt","Short Long Term Debt","Short/Long Term Debt","Total Liab"])
                equity     = _first(bs.iloc[:,0], ["Total Stockholder Equity","Total Shareholder Equity","Stockholders Equity"])
                if pd.notna(total_debt) and pd.notna(equity) and equity!=0:
                    de = float(total_debt)/float(equity)
        except: pass

    # FCF (approx) & Market Cap → FCF Yield
    fcf = info.get("freeCashflow", np.nan)
    if np.isnan(fcf):
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf = _first(cf.iloc[:,0], ["Total Cash From Operating Activities","Operating Cash Flow"])
                capex = _first(cf.iloc[:,0], ["Capital Expenditures","Capital Expenditure"])
                if pd.notna(ocf) and pd.notna(capex):
                    fcf = float(ocf) + float(capex)  # capex often negative
        except: pass
    mktcap = info.get("marketCap", np.nan)
    fcf_yield_pct = (fcf/mktcap*100) if (pd.notna(fcf) and pd.notna(mktcap) and mktcap>0) else np.nan

    return {
        "Ticker": ticker,
        "Company": name,
        "Sector": info.get("sector",""),
        "Industry": info.get("industry",""),
        "Price": price,
        "PEG Ratio": peg,
        "Revenue Growth YoY (%)": yoy,
        "Debt/Equity": de,
        "FCF (approx, $)": fcf,
        "Market Cap ($)": mktcap,
        "FCF Yield (%)": fcf_yield_pct
    }

# ---------- UI ----------
st.set_page_config(page_title="BullStock — Pro", layout="wide")
st.title("🐂 BullStock — Scoring, Moat & Filters (Auto)")

default_tickers = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, NFLX, UNH, ANET, ARM"
tickers_text = st.text_area("Tickers (comma/space separated)", default_tickers)
raw = re.split(r"[,\s]+", tickers_text.strip())
tickers = [_clean_ticker(t) for t in raw if _clean_ticker(t)]

st.sidebar.header("Economic Moat (defaults applied to all)")
brand   = st.sidebar.slider("Brand & Pricing", 0,10,6)
barriers= st.sidebar.slider("Barriers to Entry", 0,10,7)
switch  = st.sidebar.slider("Switching Costs", 0,10,6)
network = st.sidebar.slider("Network Effect", 0,10,6)
scale   = st.sidebar.slider("Economies of Scale", 0,10,8)
moat_score = moat_composite(brand,barriers,switch,network,scale)

st.sidebar.header("Filters (applied after scoring)")
flt_moat_rng  = st.sidebar.slider("Moat Score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_total_rng = st.sidebar.slider("Total Score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_peg    = st.sidebar.slider("Max PEG", 0.0, 5.0, 5.0, 0.1)
flt_rev    = st.sidebar.slider("Min Revenue Growth %", -50.0, 50.0, -50.0, 1.0)
flt_de     = st.sidebar.slider("Max Debt/Equity", 0.0, 3.0, 3.0, 0.1)
flt_fcfy   = st.sidebar.slider("Min FCF Yield %", -5.0, 15.0, -5.0, 0.5)
flt_price  = st.sidebar.slider("Min Price ($)", 0.0, 1000.0, 0.0, 1.0)

run = st.button("Run")

if run:
    rows = []
    prog = st.progress(0.0, text="Fetching...")
    for i, tk in enumerate(tickers):
        snap = fetch_snapshot(tk)

        rev_s  = score_revenue_growth(snap["Revenue Growth YoY (%)"])
        debt_s = score_debt_ratio(snap["Debt/Equity"])
        peg_s  = score_peg(snap["PEG Ratio"])
        fcfy_s = score_fcf_yield(snap["FCF Yield (%)"])

        total  = total_score(rev_s, debt_s, fcfy_s, peg_s, moat_score)

        moat_notes = "\n".join([
            moat_justification(snap["Company"], "1) Brand Loyalty & Pricing Power", brand),
            moat_justification(snap["Company"], "2) High Barriers to Entry", barriers),
            moat_justification(snap["Company"], "3) High Switching Costs", switch),
            moat_justification(snap["Company"], "4) Network Effect", network),
            moat_justification(snap["Company"], "5) Economies of Scale", scale),
        ])

        rows.append({
            **snap,
            "Revenue Score": rev_s,
            "Debt Score": debt_s,
            "PEG Score": peg_s,
            "FCF Yield Score": fcfy_s,
            "Moat Subscores (Brand,Barriers,Switching,Network,Scale)": f"{brand},{barriers},{switch},{network},{scale}",
            "Moat Score": moat_score,
            "Moat Notes": moat_notes,
            "Total Score": total
        })
        prog.progress((i+1)/len(tickers), text=f"Processed {i+1}/{len(tickers)}")

    df = pd.DataFrame(rows)

    filt = (
        (df["Moat Score"].between(flt_moat_rng[0], flt_moat_rng[1])) &
        (df["Total Score"].between(flt_total_rng[0], flt_total_rng[1])) &
        ((df["PEG Ratio"].fillna(9999)) <= flt_peg) &
        (df["Revenue Growth YoY (%)"].fillna(-9999) >= flt_rev) &
        (df["Debt/Equity"].fillna(9999) <= flt_de) &
        (df["FCF Yield (%)"].fillna(-9999) >= flt_fcfy) &
        (df["Price"].fillna(0) >= flt_price)
    )
    df_f = df[filt].reset_index(drop=True)

    st.subheader("Filtered Results")
    st.dataframe(df_f, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download FILTERED (Excel)",
            data=_excel_bytes(df_f, "Filtered"),
            file_name="bullstock_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download FILTERED (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_filtered.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download ALL (Excel)",
            data=_excel_bytes(df, "All"),
            file_name="bullstock_all.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download ALL (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_all.csv",
            mime="text/csv",
        )

    with st.expander("📘 Appendix: How BullStock Scores Each Metric"):
        st.markdown("""
**Revenue Growth YoY (0–10)**: >20%→10, >10%→8, >5%→6, >0%→4, else 0.  
**Debt/Equity (0–10, lower=better)**: <0.5→10, <1.0→7, <2.0→4, else 0.  
**PEG (0–10, lower=better)**: <1→10, <2→6, <3→3, else 0.  
**FCF Yield (0–10, higher=better)**: >8%→10, >5%→8, >3%→6, >1%→4, >0%→2, else 0.  
**Moat (5 subs, avg 0–10)**: Brand & Pricing, Barriers, Switching, Network, Scale.  
**Total Score (0–10)**: equal‑weighted avg of the five component scores.
""")
else:
    st.info("Enter tickers, adjust moat defaults & filters, then press **Run**.")
