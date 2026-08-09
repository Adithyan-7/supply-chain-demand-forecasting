import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Exploratory Data Analysis")

st.markdown("""
This section explores sales and inventory patterns in the retail dataset.
The analysis focuses on product performance, category performance,
sales trends, and inventory behavior.
""")

# Load cleaned dataset
BASE_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    BASE_DIR / "data" / "cleaned" / "retail_store_inventory_cleaned.csv",
    parse_dates=["Date"]
)
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

st.dataframe(df.head())

st.subheader("Sales Distribution")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    df["Units Sold"],
    bins=20
)

ax.set_title("Distribution of Units Sold")
ax.set_xlabel("Units Sold")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.subheader("Category-wise Sales")

category_sales = (
    df.groupby("Category")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

category_sales.plot(
    kind="bar",
    ax=ax
)

st.subheader("Top 10 Products by Sales")

top_products = (
    df.groupby("Product ID")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))

top_products.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_title("Top 10 Products by Units Sold")
ax.set_xlabel("Units Sold")
ax.set_ylabel("Product ID")

plt.tight_layout()

st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 5))

top_products.sort_values().plot(
    kind="barh",
    ax=ax
)

ax.set_title("Top 10 Products by Units Sold")
ax.set_xlabel("Units Sold")
ax.set_ylabel("Product")

plt.tight_layout()

st.pyplot(fig)

st.subheader("Daily Sales Trend")

daily_sales = (
    df.groupby("Date")["Units Sold"]
      .sum()
      .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"]
)

ax.set_title("Daily Units Sold Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

import matplotlib.dates as mdates

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Monthly Sales Trend")

monthly_sales = (
    df.set_index("Date")
      .resample("ME")["Units Sold"]
      .sum()
      .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    monthly_sales["Date"],
    monthly_sales["Units Sold"],
    marker="o"
)

ax.set_title("Monthly Units Sold")
ax.set_xlabel("Month")
ax.set_ylabel("Units Sold")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Inventory Level vs Units Sold")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["Inventory Level"],
    df["Units Sold"],
    alpha=0.5
)

ax.set_title("Inventory Level vs Units Sold")
ax.set_xlabel("Inventory Level")
ax.set_ylabel("Units Sold")

plt.tight_layout()

st.pyplot(fig)

correlation = df["Inventory Level"].corr(
    df["Units Sold"]
)

st.metric(
    "Inventory-Sales Correlation",
    f"{correlation:.2f}"
)

if correlation > 0:
    st.info(
        "Inventory level and units sold show a positive relationship "
        "in the analyzed data."
    )
elif correlation < 0:
    st.info(
        "Inventory level and units sold show a negative relationship "
        "in the analyzed data."
    )
else:
    st.info(
        "Inventory level and units sold show little linear relationship "
        "in the analyzed data."
    )

st.subheader("Store-wise Sales")

store_sales = (
    df.groupby("Store ID")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

store_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Total Units Sold by Store")
ax.set_xlabel("Store ID")
ax.set_ylabel("Total Units Sold")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Region-wise Sales")

region_sales = (
    df.groupby("Region")["Units Sold"]
      .sum()
      .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

region_sales.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Total Units Sold by Region")
ax.set_xlabel("Region")
ax.set_ylabel("Total Units Sold")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.divider()

st.subheader("EDA Summary")

st.markdown("""
The exploratory analysis provides an overview of sales distribution,
product and category performance, sales trends over time, inventory
levels, and differences across stores and regions.

These observations provide the foundation for the subsequent
anomaly detection and demand forecasting analysis.
""")