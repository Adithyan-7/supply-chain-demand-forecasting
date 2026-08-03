import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore

st.set_page_config(page_title="Anomaly Detection")

st.title("🚨 Anomaly Detection")

st.write("""
This page identifies unusual sales patterns in the retail inventory dataset
using statistical anomaly detection techniques.
""")


BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    BASE_DIR / "data" / "cleaned" / "retail_store_inventory_cleaned.csv",
    parse_dates=["Date"]
)


daily_sales = (
    df.groupby("Date")["Units Sold"]
      .sum()
      .reset_index()
)

daily_sales["Date"] = daily_sales["Date"].dt.strftime("%Y-%m-%d")

st.subheader("Daily Sales")

st.dataframe(daily_sales.head())


fig, ax = plt.subplots(figsize=(10,4))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"]
)

ax.set_title("Daily Units Sold")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

st.pyplot(fig)
st.info("""
The anomaly detection models analyze this daily sales series
to identify unusually high or low sales values.
""")
st.subheader("Z-Score Anomaly Detection")
daily_sales["Z-Score"] = zscore(daily_sales["Units Sold"])

daily_sales["Z_Anomaly"] = (
    abs(daily_sales["Z-Score"]) > 3
)
zscore_count = daily_sales["Z_Anomaly"].sum()

st.metric(
    label="Z-Score Anomalies",
    value=int(zscore_count)
)
st.subheader("Detected Z-Score Anomalies")

st.dataframe(
    daily_sales[daily_sales["Z_Anomaly"]]
)
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.scatter(
    daily_sales.loc[daily_sales["Z_Anomaly"], "Date"],
    daily_sales.loc[daily_sales["Z_Anomaly"], "Units Sold"],
    color="red",
    label="Anomalies"
)

ax.set_title("Z-Score Anomaly Detection")

import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)
st.success("""
The Z-Score method identifies observations that deviate significantly
from the average daily sales.

These anomalies may indicate unusual demand spikes, inventory shortages,
promotional campaigns, or unexpected supply chain events.
""")

#IQR Anomaly Detection

st.subheader("IQR Anomaly Detection")

Q1 = daily_sales["Units Sold"].quantile(0.25)
Q3 = daily_sales["Units Sold"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

daily_sales["IQR_Anomaly"] = (
    (daily_sales["Units Sold"] < lower_bound) |
    (daily_sales["Units Sold"] > upper_bound)
)

iqr_count = daily_sales["IQR_Anomaly"].sum()

st.metric(
    label="IQR Anomalies",
    value=int(iqr_count)
)

st.subheader("Detected IQR Anomalies")

st.dataframe(
    daily_sales[daily_sales["IQR_Anomaly"]]
)

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.scatter(
    daily_sales.loc[daily_sales["IQR_Anomaly"], "Date"],
    daily_sales.loc[daily_sales["IQR_Anomaly"], "Units Sold"],
    color="orange",
    label="IQR Anomalies"
)

ax.set_title("IQR Anomaly Detection")
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
The IQR method identifies observations that fall outside the normal spread of the data.

These unusual sales values may indicate inventory shortages, unexpected demand surges,
or operational issues that require further investigation.
""")