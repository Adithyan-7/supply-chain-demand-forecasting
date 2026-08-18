import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import zscore


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Business Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Business Insights & Recommendations")

st.write("""
This page summarizes the findings from the project's data preprocessing,
EDA, anomaly detection, and demand forecasting notebooks and translates
them into practical supply-chain recommendations.
""")


# ============================================================
# LOAD CLEANED DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "cleaned"
    / "retail_store_inventory_cleaned.csv"
)

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"]
)

df["Date"] = pd.to_datetime(df["Date"])

df["Units Sold"] = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

df["Inventory Level"] = pd.to_numeric(
    df["Inventory Level"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", "Units Sold", "Category", "Region"]
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.subheader("Project Summary")

st.markdown("""
The project analyzes retail store inventory and sales data through four
main analytical stages:

1. **Data Preprocessing** — cleaning the dataset and preparing a daily
   sales time series.
2. **Exploratory Data Analysis** — examining product, category, store,
   inventory, and sales patterns.
3. **Anomaly Detection** — identifying unusual sales observations using
   Z-Score and IQR methods.
4. **Demand Forecasting** — comparing a 7-day Moving Average baseline
   with an ARIMA model to forecast future daily demand.

The results are presented through the Streamlit dashboard to support
inventory planning and supply-chain decision-making.
""")


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()
st.subheader("📊 Dataset Overview")

total_records = len(df)
total_units = df["Units Sold"].sum()
date_start = df["Date"].min()
date_end = df["Date"].max()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Records", f"{total_records:,}")

with col2:
    st.metric("Total Units Sold", f"{total_units:,.0f}")

with col3:
    st.metric("Start Date", date_start.strftime("%Y-%m-%d"))

with col4:
    st.metric("End Date", date_end.strftime("%Y-%m-%d"))

st.info("""
The preprocessing notebook found no missing values or duplicate records
and prepared the dataset for downstream analysis. The daily time series
covers 731 days from 2022-01-01 to 2024-01-01, with no missing dates.
""")


# ============================================================
# EDA INSIGHTS
# ============================================================

st.divider()
st.subheader("🔎 EDA Insights")

category_sales = (
    df.groupby("Category")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

region_sales = (
    df.groupby("Region")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

product_sales = (
    df.groupby("Product ID")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

store_sales = (
    df.groupby("Store ID")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

best_category = category_sales.index[0]
best_region = region_sales.index[0]
best_product = product_sales.index[0]
best_store = store_sales.index[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Highest-Selling Category",
        best_category
    )

    st.metric(
        "Highest-Selling Product",
        best_product
    )

with col2:
    st.metric(
        "Highest-Selling Region",
        best_region
    )

    st.metric(
        "Highest-Selling Store",
        best_store
    )

st.markdown("""
The EDA notebook examined:

- Top products by sales and inventory
- Product frequency
- Category-wise sales
- Daily, weekly, and monthly sales trends
- Monthly order counts
- Store-wise sales and inventory
- Inventory versus sales
- Sales distributions and outliers
- Correlations among numerical variables

These analyses provide the descriptive foundation for the anomaly
detection and forecasting stages.
""")

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    category_sales.index,
    category_sales.values
)

ax.set_title("Category-wise Sales")
ax.set_xlabel("Category")
ax.set_ylabel("Units Sold")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# ============================================================
# ANOMALY DETECTION INSIGHTS
# ============================================================

st.divider()
st.subheader("🚨 Anomaly Detection Insights")

st.markdown("""
The anomaly detection notebook applied two statistical techniques to
the sales data:

- **Z-Score:** observations with an absolute Z-Score greater than 3
  were classified as anomalies.
- **IQR:** observations below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR
  were classified as anomalies.

The notebook-level EDA analysis identified **236 Z-Score anomalies**
and **715 IQR anomalies** at the individual sales-record level.
""")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Notebook Z-Score Anomalies",
        "236"
    )

with col2:
    st.metric(
        "Notebook IQR Anomalies",
        "715"
    )

st.warning("""
The IQR method identified more observations than the Z-Score method.
This difference reflects the different statistical assumptions and
thresholds of the two techniques. An anomaly should be investigated
rather than automatically treated as an error.
""")

st.markdown("""
### Possible Business Causes

The anomaly-detection notebook identified several possible explanations
for unusual sales patterns:

- Seasonal demand fluctuations
- Promotional campaigns
- Supply-chain disruptions
- Inventory shortages
- Unexpected customer demand

The notebook also examined anomaly frequency over time and by product,
which can help identify recurring demand or inventory issues.
""")


# ============================================================
# DEMAND FORECASTING INSIGHTS
# ============================================================

st.divider()
st.subheader("📈 Demand Forecasting Insights")

st.markdown("""
The forecasting notebook compared two approaches using a chronological
80/20 train-test split:

- **7-Day Moving Average**
- **ARIMA(1,1,1)**

The test period was kept separate from the training period so that the
models were evaluated on unseen future observations.
""")

comparison = pd.DataFrame({
    "Model": [
        "Moving Average",
        "ARIMA"
    ],
    "MAE": [
        907.321672,
        842.128974
    ],
    "RMSE": [
        1125.053489,
        1050.343691
    ],
    "MAPE": [
        0.068942,
        0.062596
    ]
})

st.dataframe(
    comparison.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "MAPE": "{:.2%}"
    }),
    use_container_width=True
)

best_model = "ARIMA"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Model",
        best_model
    )

with col2:
    st.metric(
        "ARIMA MAPE",
        "6.26%"
    )

with col3:
    st.metric(
        "Moving Average MAPE",
        "6.89%"
    )

st.success("""
ARIMA was the better-performing model in the notebook. It achieved
lower MAE, RMSE, and MAPE than the 7-day Moving Average baseline.

ARIMA reduced MAPE from 6.89% to 6.26%, making it the stronger model
among the two evaluated approaches for this dataset.
""")

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    comparison["Model"],
    comparison["MAPE"] * 100
)

ax.set_title("Forecasting Model Comparison")
ax.set_xlabel("Model")
ax.set_ylabel("MAPE (%)")

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# TIME-SERIES PREPARATION
# ============================================================

st.divider()
st.subheader("🕒 Time-Series Insights")

daily_sales = (
    df.groupby("Date")["Units Sold"]
    .sum()
    .sort_index()
)

st.markdown("""
The preprocessing notebook converted the transaction-level dataset
into a daily sales time series. The series contains **731 daily
observations** from 2022-01-01 to 2024-01-01.

The notebook also verified that there were **0 missing dates**.
Time-series decomposition was performed using an additive model with
a 7-day period to examine observed sales, trend, weekly seasonality,
and residual noise.
""")

decomposition_summary = pd.DataFrame({
    "Component": [
        "Observed",
        "Trend",
        "Seasonal",
        "Residual"
    ],
    "Purpose": [
        "Original daily sales",
        "Underlying long-term movement",
        "Recurring weekly pattern",
        "Irregular variation"
    ]
})

st.dataframe(
    decomposition_summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

st.divider()
st.subheader("💡 Business Recommendations")

recommendations = [
    (
        f"Prioritize inventory monitoring for **{best_category}**, "
        "which has the highest total sales among the categories in the dataset."
    ),
    (
        f"Pay close attention to **{best_region}** because it has the "
        "highest total sales among the regions."
    ),
    (
        "Investigate repeated anomalies by product and time period to "
        "distinguish genuine demand events from inventory or operational issues."
    ),
    (
        "Use the stronger ARIMA forecasting results as a baseline for "
        "procurement and replenishment planning."
    ),
    (
        "Use forecast results together with current inventory levels to "
        "reduce the risk of stock shortages and unnecessary excess inventory."
    ),
    (
        "Monitor recurring weekly and seasonal demand patterns when "
        "planning inventory and supplier orders."
    ),
    (
        "Review high-inventory products alongside their sales performance "
        "to identify potentially slow-moving stock."
    ),
    (
        "Combine analytical anomaly alerts with information about "
        "promotions, holidays, supplier delays, and stock availability."
    )
]

for i, recommendation in enumerate(
    recommendations,
    start=1
):
    st.markdown(
        f"**{i}.** {recommendation}"
    )


# ============================================================
# LIMITATIONS
# ============================================================

st.divider()
st.subheader("⚠️ Limitations")

st.markdown("""
- Forecasting performance depends on historical data quality and coverage.
- External variables such as promotions, weather, competitor behavior,
  and economic conditions were not directly modeled in ARIMA.
- Statistical anomaly detection can classify legitimate unusual
  business events as anomalies.
- The forecasting notebook compares only a 7-day Moving Average and
  ARIMA(1,1,1).
- Forecast results should be interpreted together with operational
  business knowledge.
""")


# ============================================================
# CONCLUSION
# ============================================================

st.subheader("Conclusion")

st.success(f"""
The project combines data preprocessing, EDA, statistical anomaly
detection, and time-series forecasting to support supply-chain
decision-making. The analysis identified {best_category} as the
highest-selling category and {best_region} as the highest-selling
region in the available data. The anomaly analysis detected unusual
sales observations using both Z-Score and IQR methods, while ARIMA
outperformed the 7-day Moving Average baseline with a MAPE of 6.26%.

Together, these results provide a foundation for inventory monitoring,
replenishment planning, anomaly investigation, and data-driven demand
planning.
""")
