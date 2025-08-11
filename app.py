import streamlit as st
import pandas as pd
import yfinance as yf
from io import BytesIO

# ===== PAGE CONFIG =====
st.set_page_config(page_title="BullStock", layout="wide")
st.title("🐂 BullStock – Economic Moat & Financial Screener")

# ===== DOWNLOAD FUNCTION =====
def _excel_bytes(df, sheet_name="Sheet1"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buf.seek(0)
    return buf.getvalue()

# ===== SCORING LOGIC =====
def score_moat(revenue_growth, debt_equity, peg, fcf):
    scores = {}
    justifications = {}

    # Revenue Growth
    if revenue_growth > 0.15:
        scores["Revenue Growth"] = 5
        justifications["Revenue Growth"] = "Excellent growth above 15%."
    elif revenue_growth > 0.10:
        scores["Revenue Growth"] = 4
        justifications["Revenue Growth"] = "Strong growth between 10% and 15%."
    elif revenue_growth > 0.05:
        scores["Revenue Growth"] = 3
        justifications["Revenue Growth"] = "Moderate growth between 5% and 10%."
    elif revenue_growth > 0:
        scores["Revenue Growth"] = 2
        justifications["Revenue Growth"] = "Weak growth below 5%."
    else:
        scores["Revenue Growth"] = 1
        justifications["Revenue Growth"] = "Negative growth."

    # Debt to Equity
    if debt_equity < 0.5:
        scores["Debt/Equity"] = 5
        justifications["Debt/Equity"] = "Very low leverage, strong balance sheet."
    elif debt_equity < 1:
        scores["Debt/Equity"] = 4
        justifications["Debt/Equity"] = "Moderate leverage."
    elif debt_equity < 2:
        scores["Debt/Equity"] = 3
        justifications["Debt/Equity"] = "Higher leverage but manageable."
    else:
        scores["Debt/Equity"] = 1
        justifications["Debt/Equity"] = "High leverage, risk to stability."

    # PEG Ratio
    if peg < 1:
        scores["PEG"] = 5
        justifications["PEG"] = "Undervalued relative to growth."
    elif peg < 2:
        scores["PEG"] = 4
        justifications["PEG"] = "Fairly valued relative to growth."
    elif peg < 3:
        scores["PEG"] = 3
        justifications["PEG"] = "Slightly overvalued."
    else:
        scores["PEG"] = 1
        justifications["PEG"] = "Highly overvalued."

    # Free Cash Flow
    if fcf > 0:
        scores["FCF"] = 5
        justifications["FCF"] = "Positive free cash flow."
    else:
        scores["FCF"] = 1
        justifications["FCF"] = "Negative free cash flow."

    total_score = sum(scores.values())
    return total_score, scores, justifications

# ===== FETCH FINANCIAL DATA =====
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        revenue_growth = info.get("revenueGrowth", 0) or 0
        debt_equity = info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else 0
        peg = info.get("pegRatio", 0) or 0
        fcf = info.get("freeCashflow", 0) or 0
        price = info.get("currentPrice", None)

        total_score, scores, justifications = score_moat(revenue_growth, debt_equity, peg, fcf)

        data = {
            "Ticker": ticker,
            "Price": price,
            "Revenue Growth": revenue_growth,
            "Debt/Equity": debt_equity,
            "PEG": peg,
            "Free Cash Flow": fcf,
            "Total Score": total_score,
            **scores
        }
        return data, justifications
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None

# ===== USER INPUT =====
tickers_text = st.text_area("Enter stock tickers (comma separated)", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

# ===== PROCESS DATA =====
results = []
all_justifications = {}
for t in tickers:
    row, justs = get_stock_data(t)
    if row:
        results.append(row)
        all_justifications[t] = justs

if results:
    df = pd.DataFrame(results)

    # ===== FILTERS =====
    st.subheader("Filter Options")
    min_score = st.slider("Minimum Total Score", 0, 20, 0)
    filtered_df = df[df["Total Score"] >= min_score]

    st.subheader("Results")
    st.dataframe(filtered_df)

    # ===== DOWNLOAD BUTTON =====
    st.download_button(
        label="Download Filtered Results as Excel",
        data=_excel_bytes(filtered_df),
        file_name="bullstock_filtered.xlsx"
    )

    # ===== APPENDIX =====
    st.subheader("Appendix – Scoring Justifications")
    for ticker, justs in all_justifications.items():
        st.markdown(f"### {ticker}")
        for metric, reason in justs.items():
            st.markdown(f"- **{metric}**: {reason}")

else:
    st.warning("No data to display. Please check tickers and try again.")
