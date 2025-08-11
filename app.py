# ============================== BullStock (Streamlit) ==============================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
from io import BytesIO

st.set_page_config(page_title="BullStock", layout="wide")
st.title("BullStock — Moat + Financial Screener (S&P 500)")

# ---- helpers ---------------------------------------------------------------------
def excel_bytes(df: pd.DataFrame, sheet="Sheet1") -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0); return buf.getvalue()

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

def _eps_cagr_from_financials(tkr: yf.Ticker, info: dict) -> float | None:
    try:
        inc = tkr.financials
        if inc is None or inc.empty: return None
        eps_rows = [r for r in inc.index if "eps" in r.lower()]
        if eps_rows:
            s = inc.loc[eps_rows[0]].dropna()
            if len(s) >= 2:
                v0, vN, n = float(s.iloc[-1]), float(s.iloc[0]), len(s)-1
                if v0 > 0 and vN > 0 and n > 0: return (vN/v0)**(1/n) - 1
        ni_rows = [r for r in inc.index if "net income" in r.lower()]
        so = info.get("sharesOutstanding")
        if ni_rows and so and so > 0:
            s = inc.loc[ni_rows[0]].dropna()
            if len(s) >= 2:
                v0, vN, n = float(s.iloc[-1])/float(so), float(s.iloc[0])/float(so), len(s)-1
                if v0 > 0 and vN > 0 and n > 0: return (vN/v0)**(1/n) - 1
    except: pass
    return None

def _fallback_peg(tkr: yf.Ticker, price: float | None, info: dict) -> float | None:
    pe = info.get("trailingPE")
    if pe is None and price is not None:
        fwd_eps = info.get("forwardEps")
        if fwd_eps:
            try: pe = float(price)/float(fwd_eps)
            except: pe = None
    g = _eps_cagr_from_financials(tkr, info)
    if pe and g and g > 0:
        try: return float(pe)/(float(g)*100.0)
        except: return None
    return None

@st.cache_data
def load_sp500() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            cols = {c.lower(): c for c in t.columns}
            if "symbol" in cols and ("security" in cols or "company" in cols):
                sym = cols["symbol"]; name = cols.get("security", cols.get("company"))
                w = t[[sym, name]].rename(columns={sym:"Ticker", name:"Company"})
                w["Ticker"] = w["Ticker"].str.upper().str.strip()
                w["Company"] = w["Company"].str.strip()
                return w[["Ticker","Company"]]
    except: pass
    return pd.DataFrame([
        ("AAPL","Apple Inc."), ("MSFT","Microsoft Corporation"), ("NVDA","NVIDIA Corporation"),
        ("AMZN","Amazon.com, Inc."), ("META","Meta Platforms, Inc."), ("GOOGL","Alphabet Inc. (Class A)"),
        ("UNH","UnitedHealth Group Incorporated"), ("ANET","Arista Networks, Inc."),
        ("ARM","Arm Holdings plc"), ("NFLX","Netflix, Inc."), ("PLTR","Palantir Technologies Inc.")
    ], columns=["Ticker","Company"])

SP500 = load_sp500()
_ALL_TICKERS = set(SP500["Ticker"])

def name_to_ticker(user_text: str) -> str | None:
    if not user_text: return None
    raw = user_text.strip(); cleaned = clean_tok(raw)
    if cleaned in _ALL_TICKERS: return cleaned
    if 1 <= len(cleaned) <= 6 and re.fullmatch(r"[A-Z0-9.\-]{1,6}", cleaned): return cleaned
    cand = SP500[SP500["Company"].str.upper().str.contains(re.escape(raw.upper()))]
    if not cand.empty: return cand.iloc[0]["Ticker"]
    return cleaned or None

def expand_input_to_tickers(text: str) -> list[str]:
    if not (text or "").strip(): return []
    toks = re.split(r"[,\n;| ]+", text.strip())
    out=[]; seen=set(); dedup=[]
    for t in toks:
        tk = name_to_ticker(t)
        if tk: out.append(tk)
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

# ---- scoring --------------------------------------------------------------------
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
    w = [1.0, 1.25, 1.0, 0.75, 1.25]
    v = [brand, barriers, switching, network, scale]
    return round(sum(vi*wi for vi,wi in zip(v,w))/sum(w), 2)

def moat_note(name, label, score):
    if score >= 9: nuance = ("Dominant scale and regulatory/contract entrenchment; data & distribution flywheels support pricing power.")
    elif score >= 7: nuance = ("Defendable advantages (cost, data, compliance, distribution). Replication is hard but possible.")
    elif score >= 5: nuance = ("Some advantages but credible substitutes; excess returns depend on execution.")
    elif score >= 3: nuance = ("Modest entry frictions; customer/supplier power limits pricing.")
    else: nuance = ("Commodity dynamics; durable excess returns unlikely.")
    return f"{label}: {score}/10 — {nuance}"

def total_score(rev_s, de_s, fcfy_s, peg_s, moat_s):
    return round((rev_s + de_s + fcfy_s + peg_s + moat_s) / 5.0, 2)

def first_val(series, keys):
    for k in keys:
        if k in series and pd.notna(series[k]): return series[k]
    return None

def fetch_snapshot(tk: str, moat_inputs: dict):
    t = yf.Ticker(tk)
    info = t.info or {}
    name = info.get("longName") or info.get("shortName") or tk
    price = safe_price(t)

    peg = info.get("pegRatio")
    if peg is None or pd.isna(peg): peg = _fallback_peg(t, price, info)

    yoy = None
    if info.get("revenueGrowth") is not None:
        try:
            rg = float(info["revenueGrowth"]); yoy = rg*100 if abs(rg) <= 1 else rg
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

    de = info.get("debtToEquity")
    try:
        if de is not None:
            de = float(de)
            if de > 10: de = de/100.0
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

    fcf = info.get("freeCashflow")
    if fcf is None:
        try:
            cf = t.cashflow
            if cf is not None and not cf.empty:
                ocf  = first_val(cf.iloc[:,0], ["Total Cash From Operating Activities","Operating Cash Flow"])
                capx = first_val(cf.iloc[:,0], ["Capital Expenditures","Capital Expenditure"])
                if ocf is not None and capx is not None:
                    fcf = float(ocf) + float(capx)
        except: pass
    mktcap = info.get("marketCap")
    fcfy = (float(fcf)/float(mktcap)*100.0) if (fcf and mktcap and mktcap>0) else None

    rev_s, de_s, peg_s, fcfy_s = s_rev(yoy), s_de(de), s_peg(peg), s_fcfy(fcfy)
    b,ba,sw,ne,sc = (moat_inputs[k] for k in ["brand","barriers","switching","network","scale"])
    moat_s = moat_avg(b,ba,sw,ne,sc)
    notes = "\n".join([
        moat_note(name,"1) Brand & Pricing", b),
        moat_note(name,"2) Barriers to Entry", ba),
        moat_note(name,"3) Switching Costs", sw),
        moat_note(name,"4) Network Effect", ne),
        moat_note(name,"5) Economies of Scale", sc),
    ])
    total = total_score(rev_s, de_s, fcfy_s, peg_s, moat_s)

    return {
        "Ticker": tk, "Company": name, "Price": price,
        "Revenue Growth YoY (%)": yoy, "Debt/Equity": de, "PEG Ratio": peg,
        "FCF ($)": fcf, "Market Cap ($)": mktcap, "FCF Yield (%)": fcfy,
        "Revenue Score": rev_s, "Debt Score": de_s, "PEG Score": peg_s, "FCF Yield Score": fcfy_s,
        "Moat Subscores": f"{b},{ba},{sw},{ne},{sc}", "Moat Score": moat_s, "Moat Notes": notes,
        "Total Score": total, "Profile": f"https://finance.yahoo.com/quote/{tk}/profile"
    }

# ---- sidebar --------------------------------------------------------------------
st.sidebar.header("Economic Moat (subscores 0–10)")
brand     = st.sidebar.slider("1) Brand & Pricing",    0, 10, 6)
barriers  = st.sidebar.slider("2) Barriers to Entry",  0, 10, 7)
switching = st.sidebar.slider("3) Switching Costs",    0, 10, 6)
network   = st.sidebar.slider("4) Network Effect",     0, 10, 6)
scale     = st.sidebar.slider("5) Economies of Scale", 0, 10, 8)
MOAT_INPUTS = dict(brand=brand, barriers=barriers, switching=switching, network=network, scale=scale)

st.sidebar.header("Filters (tick to enable)")
mode = st.sidebar.radio("Combine filters using", ["AND", "OR"], horizontal=True)
en_moat  = st.sidebar.checkbox("Filter by Moat score range", False)
if en_moat:  flt_moat  = st.sidebar.slider("Moat score range", 0.0, 10.0, (8.0, 10.0), 0.5)
en_total = st.sidebar.checkbox("Filter by Total score range", False)
if en_total: flt_total = st.sidebar.slider("Total score range", 0.0, 10.0, (7.0, 10.0), 0.5)
en_rev   = st.sidebar.checkbox("Filter by Min Revenue Growth %", False)
if en_rev:   flt_rev   = st.sidebar.slider("Min Revenue Growth %", -50.0, 50.0, 10.0, 1.0)
en_peg   = st.sidebar.checkbox("Filter by Max PEG", False)
if en_peg:   flt_peg   = st.sidebar.slider("Max PEG", 0.0, 10.0, 2.0, 0.1)
en_de    = st.sidebar.checkbox("Filter by Max Debt/Equity", False)
if en_de:    flt_de    = st.sidebar.slider("Max Debt/Equity", 0.0, 5.0, 1.0, 0.1)
en_fcfy  = st.sidebar.checkbox("Filter by Min FCF Yield %", False)
if en_fcfy:  flt_fcfy  = st.sidebar.slider("Min FCF Yield %", -10.0, 20.0, 0.0, 0.5)
en_price = st.sidebar.checkbox("Filter by Min Price ($)", False)
if en_price: flt_price = st.sidebar.slider("Min Price ($)", 0.0, 2000.0, 0.0, 1.0)

st.sidebar.divider()
st.sidebar.header("Screener mode (S&P 500)")
scan_all = st.sidebar.checkbox("Scan entire S&P 500", value=False)
sc_mode = st.sidebar.radio("Combine screener rules with", ["AND", "OR"], horizontal=True)
en_sc_moat = st.sidebar.checkbox("Enable: Min Moat", True)
if en_sc_moat: sc_min_moat = st.sidebar.slider("Min Moat", 0.0, 10.0, 8.0, 0.5)
en_sc_rev = st.sidebar.checkbox("Enable: Min Revenue Growth %", False)
if en_sc_rev: sc_min_rev = st.sidebar.number_input("Min Revenue Growth %", value=23.0, step=1.0)
en_sc_peg = st.sidebar.checkbox("Enable: Max PEG", False)
if en_sc_peg: sc_max_peg = st.sidebar.number_input("Max PEG", value=5.0, step=0.1, min_value=0.0)
en_sc_de = st.sidebar.checkbox("Enable: Max Debt/Equity", False)
if en_sc_de: sc_max_de = st.sidebar.number_input("Max Debt/Equity", value=3.0, step=0.1, min_value=0.0)
en_sc_fcfy = st.sidebar.checkbox("Enable: Min FCF Yield %", False)
if en_sc_fcfy: sc_min_fcfy = st.sidebar.number_input("Min FCF Yield %", value=-5.0, step=0.5)

# ---- main input -----------------------------------------------------------------
defaults = "AAPL, MSFT, NVDA, AMZN, META, GOOGL, UNH, ANET, ARM, NFLX, PLTR"
user_text = st.text_area("Enter tickers or company names (comma/space/newline)", defaults)

typed = expand_input_to_tickers(user_text or defaults)
universe = list(SP500["Ticker"]) if scan_all else typed
if scan_all:
    universe = sorted(set(universe).union(set(typed)))

run = st.button("Run", type="primary")

# ---- execution ------------------------------------------------------------------
if run:
    if not universe:
        st.warning("No valid tickers found. Try names (e.g., 'Palantir') or enable S&P scan.")

    rows=[]; prog = st.progress(0.0, text="Fetching...")
    for i, tk in enumerate(universe):
        try: rows.append(fetch_snapshot(tk, MOAT_INPUTS))
        except Exception as e: st.warning(f"Fetch failed for {tk}: {e}")
        prog.progress((i+1)/max(len(universe),1), text=f"Processed {i+1}/{len(universe)}")

    df = pd.DataFrame(rows)

    # Screener pre-filter (AND/OR with enable flags)
    screener_ok = pd.Series(True, index=df.index)
    if scan_all:
        sc_conds = []
        if en_sc_moat: sc_conds.append(df["Moat Score"].fillna(0) >= float(sc_min_moat))
        if en_sc_rev:  sc_conds.append(df["Revenue Growth YoY (%)"].fillna(-1e9) >= float(sc_min_rev))
        if en_sc_peg:  sc_conds.append(df["PEG Ratio"].fillna(1e9) <= float(sc_max_peg))
        if en_sc_de:   sc_conds.append(df["Debt/Equity"].fillna(1e9) <= float(sc_max_de))
        if en_sc_fcfy: sc_conds.append(df["FCF Yield (%)"].fillna(-1e9) >= float(sc_min_fcfy))
        if sc_conds:
            screener_ok = sc_conds[0]
            for c in sc_conds[1:]:
                screener_ok = (screener_ok & c) if sc_mode == "AND" else (screener_ok | c)

    # UI checkbox filters (AND/OR)
    conds=[]
    if en_moat:   conds.append(df["Moat Score"].fillna(0).between(flt_moat[0], flt_moat[1]))
    if en_total:  conds.append(df["Total Score"].fillna(0).between(flt_total[0], flt_total[1]))
    if en_rev:    conds.append(df["Revenue Growth YoY (%)"].fillna(-1e9) >= flt_rev)
    if en_peg:    conds.append(df["PEG Ratio"].fillna(1e9) <= flt_peg)
    if en_de:     conds.append(df["Debt/Equity"].fillna(1e9) <= flt_de)
    if en_fcfy:   conds.append(df["FCF Yield (%)"].fillna(-1e9) >= flt_fcfy)
    if en_price:  conds.append(df["Price"].fillna(-1e9) >= flt_price)
    if not conds:
        cond = pd.Series(True, index=df.index)
    else:
        cond = conds[0]
        for c in conds[1:]:
            cond = (cond & c) if mode == "AND" else (cond | c)

    filt = (screener_ok if scan_all else pd.Series(True, index=df.index)) & cond
    df_f = df[filt].reset_index(drop=True)

    # Display + downloads
    df_f.insert(1, "Profile Link", df_f["Profile"])
    colcfg = {"Profile Link": st.column_config.LinkColumn("Ticker (link)", display_text="→"),
              "Profile": None}
    st.subheader("Filtered Results")
    st.dataframe(df_f.drop(columns=[c for c in ["Profile"] if c in df_f.columns]),
                 use_container_width=True, height=520, column_config=colcfg)
    st.caption(f"Showing {len(df_f)} of {len(df)} rows")

    c1,c2 = st.columns(2)
    with c1:
        st.download_button("Download FILTERED (Excel)", excel_bytes(df_f,"Filtered"),
                           "bullstock_filtered.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download FILTERED (CSV)", df_f.to_csv(index=False).encode("utf-8"),
                           "bullstock_filtered.csv","text/csv")
    with c2:
        st.download_button("Download ALL (Excel)", excel_bytes(df,"All"),
                           "bullstock_all.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download ALL (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "bullstock_all.csv","text/csv")

    with st.expander("Appendix: Scoring Rules"):
        st.markdown(
            "- Revenue Growth YoY: >20%→10, >10%→8, >5%→6, >0%→4.\n"
            "- Debt/Equity (lower better): <0.5→10, <1.0→7, <2.0→4.\n"
            "- PEG (lower better): <1→10, <2→6, <3→3; missing PEG = neutral 5 (fallback calc used).\n"
            "- FCF Yield (higher better): >8%→10, >5%→8, >3%→6, >1%→4, >0%→2.\n"
            "- Moat: weighted avg (Brand 1.0, Barriers 1.25, Switching 1.0, Network 0.75, Scale 1.25)."
        )
else:
    st.info("Type names/tickers (e.g., ‘Palantir’ or ‘PLTR’) or enable ‘Scan entire S&P 500’, then press Run.")
