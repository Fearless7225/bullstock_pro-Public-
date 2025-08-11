import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO

st.set_page_config(page_title="BullStock", layout="wide")
st.title("BullStock — Moat + Financial Screener")

# ---------- helpers ----------
def excel_bytes(df: pd.DataFrame, sheet="Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf.getvalue()

def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\.\-]", "", (t or "").upper())

def safe_price(tkr: yf.Ticker):
    info = tkr.info or {}
    p = info.get("currentPrice")
    if p:
        return float(p)
    try:
        h = tkr.history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return np.nan

# ---------- scoring (0–10, missing -> 0) ----------
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
    if p < 1: return 10
    if p < 2: return 6
    if p < 3: return 3
    return 0

def s_fcfy(y):
    if y is None or pd.isna(y): return 0
    if y > 8: return 10
    if y > 5: return 8
    if y > 3: return 6
    if y > 1: return 4
    if y > 0: return 2
    return 0

def moat_avg(b, r, s, n, e):
    return round((b + r + s + n + e) / 5.0, 2)

def moat_note(name, label, score):
    if score >= 9: txt = "Exceptional and durable."
    elif score >= 7: txt = "Strong but not absolute."
    elif score >= 5: txt = "Moderate; contested."
    elif score >= 3: txt = "Weak; easily matched."
    else: txt = "Minimal; commoditized."
    return f"{label}: {score}/10 — {txt} ({name})."

def total_score(rev_s, de_s, fcfy_s, peg_s, moat_s):
    scores = [rev_s, de_s, fcfy_s, peg_s, moat_s]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    num = sum(s*w for s, w in zip(scores, weights) if s is not None and not pd.isna(s))
    den = sum(w for s, w in zip(scores, weights) if s is not None and not pd.isna(s))
    return round(num / den, 2) if den > 0 else 0.0

# ---------- data fetch ----------
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

    # revenue growth YoY (%)
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
                    if len(rev) >= 2:
                        yoy = float((rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100)
        except Exception:
            pass

    # debt/equity
    de = info.get("debtToEquity")
    try:
        if de is not None and de > 10:
            de = de / 100.0
    except Exception:
        pass
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

    # FCF yield
    fcf = info.get("freeCashflow")
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf = first_val(cf.iloc[:, 0], ["Total Cash From Operating Activities", "Operating Cash Flow"])
                capex = first_val(cf.iloc[:, 0], ["Capital Expenditures", "Capital Expenditure"])
                if ocf is not None and capex is not None:
                    fcf = float(ocf) + float(capex)  # capex usually negative
        except Exception:
            pass
    mktcap = info.get("marketCap")
    fcfy = (fcf / mktcap * 100.0) if (fcf and mktcap and mktcap > 0) else None

    return {
        "Ticker": tk, "Company": name, "Price": price,
        "Revenue Growth YoY (%)": yoy, "Debt/Equity": de,
        "PEG Ratio": peg, "FCF ($)": fcf, "Market Cap ($)": mktcap,
        "FCF Yield (%)": fcfy
    }

# ---------- UI ----------
defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX"
tickers_text = st.text_area("Tickers (comma or space separated)", defaults)
tickers = [clean_ticker(t) for t in re.split(r"[,\s]+", tickers_text.strip()) if t]

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

run = st.button("Run")

if run:
    rows = []
    prog = st.progress(0.0, text="Fetching...")
    for i, tk in enumerate(tickers):
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
        prog.progress((i + 1) / len(tickers), text=f"Processed {i + 1}/{len(tickers)}")

    df = pd.DataFrame(rows)

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
            "Revenue Growth YoY (0–10): >20%→10, >10%→8, >5%→6, >0%→4, else 0.\n"
            "Debt/Equity (0–10, lower better): <0.5→10, <1.0→7, <2.0→4, else 0.\n"
            "PEG (0–10, lower better): <1→10, <2→6, <3→3, else 0.\n"
            "FCF Yield (0–10, higher better): >8%→10, >5%→8, >3%→6, >1%→4, >0%→2, else 0.\n"
            "Moat (avg of 5 subs: Brand & Pricing, Barriers, Switching, Network, Scale).\n"
            "Total Score: equal-weighted average of the four metrics plus moat."
        )
else:
    st.info("Enter tickers, set moat subscores on the left, then press Run.")
