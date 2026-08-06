import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Demand Forecasting")

st.title("📈 Demand Forecasting")

st.write("""
This page presents demand forecasting results using
Moving Average and ARIMA models.
""")

BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    BASE_DIR / "data" / "cleaned" / "retail_store_inventory_cleaned.csv",
    parse_dates=["Date"]
)

#Daily sales

daily_sales = (
    df.groupby("Date")["Units Sold"]
      .sum()
      .reset_index()
)
daily_sales["Date"] = daily_sales["Date"].dt.strftime("%Y-%m-%d")

st.subheader("Daily Sales")

st.dataframe(daily_sales.head())

#Plot daily sales

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"]
)

ax.set_title("Daily Sales Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

st.pyplot(fig)

st.info("""
The daily sales time series is used to train forecasting models
that predict future demand.

This helps businesses improve inventory planning and optimize
supply chain operations.
""")