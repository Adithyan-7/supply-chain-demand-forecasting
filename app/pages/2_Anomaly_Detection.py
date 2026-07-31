import streamlit as st

st.set_page_config(page_title="Anomaly Detection")

st.title("🚨 Anomaly Detection")

st.write("""
This page identifies unusual sales patterns in the retail inventory dataset
using statistical anomaly detection techniques.
""")
import pandas as pd

df = pd.read_csv("../data/cleaned/retail_store_inventory_cleaned.csv")
daily_sales = (
    df.groupby("Date")["Units Sold"]
      .sum()
      .reset_index()
)
st.subheader("Daily Sales")

st.dataframe(daily_sales.head())
import matplotlib.pyplot as plt

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