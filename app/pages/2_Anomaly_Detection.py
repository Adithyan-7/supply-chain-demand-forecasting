import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from scipy.stats import zscore


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Anomaly Detection")

st.write("""
This page identifies unusual sales and demand patterns in the retail
inventory dataset using statistical anomaly detection techniques.
Z-Score and IQR methods are used to identify unusually high or low
daily sales observations.
""")


# ============================================================
# LOAD DATA
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


# ============================================================
# DATA PREPARATION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Units Sold"] = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", "Category", "Units Sold"]
)


# ============================================================
# CATEGORY SELECTION
# ============================================================

st.subheader("🎯 Select Product Category")

categories = sorted(
    df["Category"].dropna().unique()
)

selected_category = st.selectbox(
    "Choose a category:",
    categories
)

category_df = df[
    df["Category"] == selected_category
].copy()


# ============================================================
# DAILY SALES
# ============================================================

daily_sales = (
    category_df
    .groupby("Date")["Units Sold"]
    .sum()
    .reset_index()
    .sort_values("Date")
)


# Keep Date as datetime
daily_sales["Date"] = pd.to_datetime(
    daily_sales["Date"]
)


# ============================================================
# DAILY SALES OVERVIEW
# ============================================================

st.subheader(
    f"📊 Daily Sales — {selected_category}"
)

st.dataframe(
    daily_sales.head(10),
    use_container_width=True
)


# ============================================================
# DAILY SALES CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.set_title(
    f"Daily Units Sold — {selected_category}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

ax.xaxis.set_major_locator(
    mdates.MonthLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)

st.info("""
The daily sales series is used as the basis for statistical anomaly
detection. Unusually high or low sales values are investigated using
Z-Score and IQR methods.
""")


# ============================================================
# Z-SCORE ANOMALY DETECTION
# ============================================================

st.divider()

st.subheader("📐 Z-Score Anomaly Detection")

daily_sales["Z-Score"] = zscore(
    daily_sales["Units Sold"]
)

daily_sales["Z_Anomaly"] = (
    daily_sales["Z-Score"].abs() > 3
)

zscore_count = int(
    daily_sales["Z_Anomaly"].sum()
)


# ============================================================
# Z-SCORE METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Z-Score Anomalies",
        zscore_count
    )

with col2:
    st.metric(
        "Z-Score Threshold",
        "±3"
    )

with col3:
    st.metric(
        "Total Observations",
        len(daily_sales)
    )


# ============================================================
# Z-SCORE ANOMALY TABLE
# ============================================================

st.subheader("Detected Z-Score Anomalies")

zscore_anomalies = daily_sales[
    daily_sales["Z_Anomaly"]
].copy()

st.dataframe(
    zscore_anomalies[
        [
            "Date",
            "Units Sold",
            "Z-Score"
        ]
    ],
    use_container_width=True
)


# ============================================================
# Z-SCORE VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.scatter(
    daily_sales.loc[
        daily_sales["Z_Anomaly"],
        "Date"
    ],
    daily_sales.loc[
        daily_sales["Z_Anomaly"],
        "Units Sold"
    ],
    color="red",
    label="Z-Score Anomalies",
    zorder=3
)

ax.set_title(
    f"Z-Score Anomaly Detection — {selected_category}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

ax.xaxis.set_major_locator(
    mdates.MonthLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)


st.success("""
The Z-Score method identifies observations that are significantly
different from the average daily sales level. Large positive values
represent unusually high sales, while large negative values represent
unusually low sales.
""")


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

st.divider()

st.subheader("📦 IQR Anomaly Detection")

Q1 = daily_sales["Units Sold"].quantile(
    0.25
)

Q3 = daily_sales["Units Sold"].quantile(
    0.75
)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)

upper_bound = Q3 + (1.5 * IQR)

daily_sales["IQR_Anomaly"] = (
    (daily_sales["Units Sold"] < lower_bound)
    |
    (daily_sales["Units Sold"] > upper_bound)
)

iqr_count = int(
    daily_sales["IQR_Anomaly"].sum()
)


# ============================================================
# IQR METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "IQR Anomalies",
        iqr_count
    )

with col2:
    st.metric(
        "Lower Bound",
        f"{lower_bound:,.0f}"
    )

with col3:
    st.metric(
        "Upper Bound",
        f"{upper_bound:,.0f}"
    )


# ============================================================
# IQR ANOMALY TABLE
# ============================================================

st.subheader("Detected IQR Anomalies")

iqr_anomalies = daily_sales[
    daily_sales["IQR_Anomaly"]
].copy()

st.dataframe(
    iqr_anomalies[
        [
            "Date",
            "Units Sold"
        ]
    ],
    use_container_width=True
)


# ============================================================
# IQR VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.scatter(
    daily_sales.loc[
        daily_sales["IQR_Anomaly"],
        "Date"
    ],
    daily_sales.loc[
        daily_sales["IQR_Anomaly"],
        "Units Sold"
    ],
    color="orange",
    label="IQR Anomalies",
    zorder=3
)

ax.axhline(
    upper_bound,
    linestyle="--",
    label="Upper IQR Bound"
)

ax.axhline(
    lower_bound,
    linestyle="--",
    label="Lower IQR Bound"
)

ax.set_title(
    f"IQR Anomaly Detection — {selected_category}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

ax.xaxis.set_major_locator(
    mdates.MonthLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)


st.success("""
The IQR method identifies observations that fall outside the expected
range defined by the first and third quartiles. This method is useful
for identifying unusually high or low sales values without relying
directly on the mean and standard deviation.
""")


# ============================================================
# METHOD COMPARISON
# ============================================================

st.divider()

st.subheader("⚖️ Comparison of Detection Methods")

comparison = pd.DataFrame({
    "Method": [
        "Z-Score",
        "IQR"
    ],
    "Anomalies Detected": [
        zscore_count,
        iqr_count
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)


fig, ax = plt.subplots(
    figsize=(7, 4)
)

ax.bar(
    comparison["Method"],
    comparison["Anomalies Detected"]
)

ax.set_title(
    "Comparison of Anomaly Detection Methods"
)

ax.set_xlabel("Method")
ax.set_ylabel("Number of Anomalies")

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# COMBINED ANOMALY ANALYSIS
# ============================================================

daily_sales["Both_Methods"] = (
    daily_sales["Z_Anomaly"]
    &
    daily_sales["IQR_Anomaly"]
)

daily_sales["Either_Method"] = (
    daily_sales["Z_Anomaly"]
    |
    daily_sales["IQR_Anomaly"]
)

both_count = int(
    daily_sales["Both_Methods"].sum()
)

either_count = int(
    daily_sales["Either_Method"].sum()
)


# ============================================================
# SUMMARY METRICS
# ============================================================

st.subheader("📋 Anomaly Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Z-Score",
        zscore_count
    )

with col2:
    st.metric(
        "IQR",
        iqr_count
    )

with col3:
    st.metric(
        "Detected by Both",
        both_count
    )

with col4:
    st.metric(
        "Detected by Either",
        either_count
    )


# ============================================================
# COMBINED HISTORICAL ANOMALY VIEW
# ============================================================

st.subheader(
    "🚨 Historical Sales Anomalies"
)

fig, ax = plt.subplots(
    figsize=(14, 6)
)

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

# Z-score anomalies
ax.scatter(
    daily_sales.loc[
        daily_sales["Z_Anomaly"],
        "Date"
    ],
    daily_sales.loc[
        daily_sales["Z_Anomaly"],
        "Units Sold"
    ],
    color="red",
    label="Z-Score",
    zorder=3
)

# IQR anomalies
ax.scatter(
    daily_sales.loc[
        daily_sales["IQR_Anomaly"],
        "Date"
    ],
    daily_sales.loc[
        daily_sales["IQR_Anomaly"],
        "Units Sold"
    ],
    color="orange",
    marker="x",
    label="IQR",
    zorder=4
)

ax.set_title(
    f"Historical Sales Anomalies — {selected_category}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")

ax.xaxis.set_major_locator(
    mdates.MonthLocator()
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)
plt.tight_layout()

ax.legend()

st.pyplot(fig)


# ============================================================
# COMBINED ANOMALY TABLE
# ============================================================

st.subheader("Detected Anomaly Events")

anomaly_events = daily_sales[
    daily_sales["Either_Method"]
].copy()

anomaly_events["Detection Method"] = np.select(
    [
        anomaly_events["Both_Methods"],
        anomaly_events["Z_Anomaly"],
        anomaly_events["IQR_Anomaly"]
    ],
    [
        "Both Z-Score and IQR",
        "Z-Score",
        "IQR"
    ],
    default="Unknown"
)

st.dataframe(
    anomaly_events[
        [
            "Date",
            "Units Sold",
            "Z-Score",
            "Detection Method"
        ]
    ],
    use_container_width=True
)


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.divider()

st.subheader("💡 Business Interpretation")

st.markdown("""
An anomaly represents an observation that differs significantly from
the normal sales pattern. It does not automatically indicate an error.

Potential reasons for unusually high sales may include:

- Promotional campaigns
- Holiday periods
- Seasonal demand
- Bulk purchases
- Unexpected increases in customer demand

Potential reasons for unusually low sales may include:

- Reduced customer demand
- Inventory availability issues
- Supplier delays
- Operational problems
- Unexpected market conditions

The detected events should therefore be investigated together with
business and operational information before taking corrective action.
""")


# ============================================================
# CONCLUSION
# ============================================================

st.subheader("Conclusion")

st.success(f"""
For the selected **{selected_category}** category, the anomaly detection
analysis identified **{either_count} potentially unusual sales events**
using either the Z-Score or IQR method.

Using both statistical approaches provides a broader view of unusual
demand behavior. The detected anomalies can be investigated further
using business context such as promotions, holidays, inventory
conditions and operational events.
""")