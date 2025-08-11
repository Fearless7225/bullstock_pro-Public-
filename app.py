import streamlit as st
import yfinance as yf
import pandas as pd

# ----- PAGE SETUP -----
st.set_page_config(page_title="BullStock", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("📈 BullStock - Economic Moat Scoring")

# ----- USER INPUT -----
ticker = st.text_input("Enter a stock ticker symbol (e.g., AAPL, MSFT, NVDA):", "AAPL")

# ----- FETCH STOCK DATA -----
try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ----- DISPLAY BASIC STOCK INFO -----
st.subheader(f"{info.get('longName', ticker)} ({ticker.upper()})")
st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,} | **Forward PE:** {info.get('forwardPE', 'N/A')}")

# ----- PRICE CHART -----
st.line_chart(hist['Close'], height=250)

# ----- ECONOMIC MOAT SCORING -----
st.subheader("🛡 Economic Moat Scorecard (max 10 each)")

criteria = [
    "Brand Loyalty & Pricing Power",
    "High Barriers to Entry",
    "High Switching Costs",
    "Network Effect",
    "Economies of Scale"
]

scores = {}
total_score = 0

for c in criteria:
    score = st.slider(c, 0, 10, 5)
    scores[c] = score
    total_score += score

# ----- FINAL SCORE -----
st.markdown(f"### ✅ **Total Moat Score:** {total_score} / 50")

# ----- SAVE RESULTS -----
if st.button("Save Results"):
    result_df = pd.DataFrame(list(scores.items()), columns=["Criteria", "Score"])
    result_df.loc[len(result_df.index)] = ["Total", total_score]
    result_df.to_csv(f"{ticker}_moat_score.csv", index=False)
    st.success(f"Results saved as {ticker}_moat_score.csv")

st.caption("Data source: Yahoo Finance | Scoring is manual input based on your analysis.")
