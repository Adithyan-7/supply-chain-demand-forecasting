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

st.subheader("Train-Test Split")

split_index = int(len(daily_sales) * 0.8)

train = daily_sales.iloc[:split_index]
test = daily_sales.iloc[split_index:]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Samples",
        len(train)
    )

with col2:
    st.metric(
        "Testing Samples",
        len(test)
    )

    st.subheader("Training Dataset")

st.dataframe(train.head())

st.subheader("Testing Dataset")

st.dataframe(test.head())

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    train["Date"],
    train["Units Sold"],
    label="Training Data"
)

ax.plot(
    test["Date"],
    test["Units Sold"],
    label="Testing Data"
)

ax.set_title("Chronological Train-Test Split")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)

st.success("""
The dataset is divided chronologically to preserve the temporal order of observations.

The training data is used to build forecasting models, while the testing data is reserved for evaluating how well the models predict future demand.
""")