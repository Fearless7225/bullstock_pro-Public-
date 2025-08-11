import streamlit as st
import pandas as pd
import yfinance as yf
import re
from io import BytesIO

# ========== PAGE ==========
st.set_page_config(page_title="BullStock", layout="wide")
st.title("🐂 BullStock — Moat + Financial Screener")

# ========== HELPERS ==========
def _excel_bytes(df: pd.DataFrame, sheet="Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf.getvalue()

def _clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\.\-]", "", (t or "").upper())

def _safe_price(tkr: yf.Ticker):
    info = tkr.info or {}
    p = info.get("currentPrice")
    if p: return float(p)
    try:
        h = tkr.history(period="1d")
        if not h.empty: return float(h["Close"].iloc[-1])
    except: pass
    return None

# ========== SCORING (0–10) ==========
def score_rev_growth(yoy_pct):
    if yoy_pct is None or pd.isna(yoy_pct): return 0
    if yoy_pct > 20: return 10
    if yoy_pct > 10: return 8
    if yoy_pct > 5:  return 6
    if yoy_pct > 0:  return 4
    return 0

def score_de_ratio(de_ratio):
    if de_ratio is None or pd.isna(de_ratio): return 0
    if de_ratio < 0.5: return 10
    if de_ratio < 1.0: return 7
    if de_ratio < 2.0: return 4
    return 0

def score_peg(peg):
    if peg is None or pd.isna(peg): return 0
    if peg < 1:  return 10
    if peg < 2:  return 6
    if peg < 3:  return 3
    return 0

def score_fcf_yield(fcf_y):
    if fcf_y is None or pd.isna(fcf_y): return 0
    if fcf_y > 8:  return 10
    if fcf_y > 5:  return 8
    if fcf_y > 3:  return 6
    if fcf_y > 1:  return 4
    if fcf_y > 0:  return 2
    return 0

def moat_avg(b, r, s, n, e):
    return round((b+r+s+n+e)/5, 2)

def moat_note(name, label, score):
    if score >= 9: text = "Exceptional & durable."
    elif score >= 7: text = "Strong but not absolute."
    elif score >= 5: text = "Moderate; contested."
    elif score >= 3: text = "Weak; easily matched."
    else: text = "Minimal; commoditized."
    return f"{label}: {score}/10 — {text} ({name})."

def total_score(rev_s, de_s, fcfy_s, peg_s, moat_s,
                w=0.2):  # equal weights
    parts = [rev_s, de_s, fcfy_s, peg_s, moat_s]
    wts   = [w, w, w, w, w]
    num = sum(p*w for p,w in zip(parts,wts) if p is not None)
    den = sum(w for p,w in zip(parts,wts) if p is not None)
    return round(num/den, 2) if den else 0.0

# ========== DATA FETCH ==========
def _first(series, keys):
    for k in keys:
        if k in series and pd.notna(series[k]): return series[k]
    return None

def fetch_metrics(tk: str):
    t = yf.Ticker(tk)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or tk

    # price
    price = _safe_price(t)

    # PEG
    peg = info.get("pegRatio")

    # revenue growth YoY (%)
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
                    if len(rev) >= 2:
                        yoy = float((rev.iloc[0]-rev.iloc[1]) / rev.iloc[1] * 100)
        except: pass

    # debt/equity
    de = info.get("debtToEquity")
    try:
        if de is not None and de > 10: de = de/100.0
    except: pass
    if de is None:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                total_debt = _first(bs.iloc[:,0], ["Total Debt","Short Long Term Debt","Short/Long Term Debt","Total Liab"])
                equity     = _first(bs.iloc[:,0], ["Total Stockholder Equity","Total Shareholder Equity","Stockholders Equity"])
                if total_debt and equity and equity != 0:
                    de = float(total_debt)/float(equity)
        except: pass

    # FCF yield
    fcf = info.get("freeCashflow")
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf = _first(cf.iloc[:,0], ["Total Cash From Operating Activities","Operating Cash Flow"])
                capex = _first(cf.iloc[:,0], ["Capital Expenditures","Capital Expenditure"])
                if ocf is not None and capex is not None:
                    fcf = float(ocf) + float(capex)   # capex negative
        except: pass
    mktcap = info.get("marketCap")
    fcf_y = (fcf/mktcap*100) if (fcf and mktcap and mktcap>0) else None

    return {
        "Ticker": tk, "Company": name, "Price": price,
        "Revenue Growth YoY (%)": yoy, "Debt/Equity": de,
        "PEG Ratio": peg, "FCF ($)": fcf, "Market Cap ($)": mktcap,
        "FCF Yield (%)": fcf_y
    }

# ========== UI ==========
defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX"
tickers_text = st.text_area("Tickers (comma/space separated)", defaults)
tickers = [_clean_ticker(t) for t in re.split(r"[,\s]+", tickers_text.strip()) if t]

# Moat sliders (apply to all rows this run)
st.sidebar.header("Economic Moat (set subscores 0–10)")
brand   = st.sidebar.slider("1) Brand & Pricing", 0,10,6)
barrier = st.sidebar.slider("2) Barriers to Entry", 0,10,7)
switch  = st.sidebar.slider("3) Switching Costs", 0,10,6)
network = st.sidebar.slider("4) Network Effect", 0,10,6)
scale   = st.sidebar.slider("5) Economies of Scale", 0,10,8)
moat_s  = moat_avg(brand, barrier, switch, network, scale)

# Filters
st.sidebar.header("Filters")
flt_moat  = st.sidebar.slider("Moat score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_total = st.sidebar.slider("Total score range", 0.0, 10.0, (0.0, 10.0), 0.5)
flt_peg   = st.sidebar.slider("Max PEG", 0.0, 5.0, 5.0, 0.1)
flt_rev   = st.sidebar.slider("Min Rev Growth %", -50.0, 50.0, -50.0, 1.0)
flt_de    = st.sidebar.slider("Max Debt/Equity", 0.0, 3.0, 3.0, 0.1)
flt_fcfy  = st.sidebar.slider("Min FCF Yield %", -5.0, 15.0, -5.0, 0.5)

run = st.button("Run")

if run:
    rows = []
    prog = st.progress(0.0, text="Fetching…")
    for i, tk in enumerate(tickers):
        m = fetch_metrics(tk)

        rev_s  = score_rev_growth(m["Revenue Growth YoY (%)"])
        de_s   = score_de_ratio(m["Debt/Equity"])
        peg_s  = score_peg(m["PEG Ratio"])
        fcfy_s = score_fcf_yield(m["FCF Yield (%)"])

        total  = total_score(rev_s, de_s, fcfy_s, peg_s, moat_s)

        moat_notes = "\n".join([
            moat_note(m["Company"], "1) Brand & Pricing", brand),
            moat_note(m["Company"], "2) Barriers to Entry", barrier),
            moat_note(m["Company"], "3) Switching Costs", switch),
            moat_note(m["Company"], "4) Network Effect", network),
            moat_note(m["Company"], "5) Economies of Scale", scale),
        ])

        rows.append({**m,
            "Revenue Score": rev_s, "Debt Score": de_s,
            "PEG Score": peg_s, "FCF Yield Score": fcfy_s,
            "Moat Subscores": f"{brand},{barrier},{switch},{network},{scale}",
            "Moat Score": moat_s, "Moat Notes": moat_notes,
            "Total Score": total
        })
        prog.progress((i+1)/len(tickers), text=f"Processed {i+1}/{len(tickers)}")

    df = pd.DataFrame(rows)

    # Apply filters
    filt = (
        df["Moat Score"].fillna(0).between(flt_moat[0], flt_moat[1]) &
        df["Total Score"].fillna(0).between(flt_total[0], flt_total[1]) &
        (df["PEG Ratio"].fillna(9999) <= flt_peg) &
        (df["Revenue Growth YoY (%)"].fillna(-9999) >= flt_rev) &
        (df["Debt/Equity"].fillna(9999) <= flt_de) &
        (df["FCF Yield (%)"].fillna(-9999) >= flt_fcfy)
    )
    df_f = df[filt].reset_index(drop=True)

    st.subheader("Filtered Results")
    st.dataframe(df_f, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download FILTERED (Excel)",
            data=_excel_bytes(df_f, "Filtered"),
            file_name="bullstock_filtered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download FILTERED (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_filtered.csv", mime="text/csv")
    with c2:
        st.download_button("Download ALL (Excel)",
            data=_excel_bytes(df, "All"),
            file_name="bullstock_all.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download ALL (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bullstock_all.csv", mime="text/csv")

    with st.expander("📘 Appendix: Scoring Rules"):
        st.markdown("""
**Revenue Growth YoY (0–10)** — >20%→10, >10%→8, >5%→6, >0%→4, else 0.  
**Debt/Equity (0–10, lower=better)** — <0.5→10, <1.0→7, <2.0→4, else 0.  
**PEG (0–10, lower=better)** — <1→10, <2→6, <3→3, else 0.  
**FCF Yield (0–10, higher=better)** — >8%→10, >5%→8, >3%→6, >1%→4, >0%→2, else 0.  
**Economic Moat (5 subs, avg 0–10)** — Brand & Pricing, Barriers, Switching, Network, Scale.  
**Moat notes** use bands: 9–10 Exceptional; 7–8 Strong; 5–6 Moderate; 3–4 Weak; 0–2 Minimal.  
**Total Score** — equal‑weighted average of the four metrics + moat.
""")

else:
    st.info("Enter tickers, set moat subscores (sidebar), then press **Run**.") on
