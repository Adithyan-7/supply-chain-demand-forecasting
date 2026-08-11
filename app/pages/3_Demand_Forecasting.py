import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Demand Forecasting")

st.write("""
This page forecasts future demand using historical sales data.
Users can select a product category and compare Moving Average
and ARIMA forecasting approaches.
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

df = df.dropna(
    subset=["Date", "Category", "Units Sold"]
)

df["Units Sold"] = pd.to_numeric(
    df["Units Sold"],
    errors="coerce"
)

df = df.dropna(
    subset=["Units Sold"]
)


# ============================================================
# CATEGORY SELECTION
# ============================================================

st.subheader("🎯 Select Product Category")

categories = sorted(
    df["Category"].dropna().unique()
)

selected_category = st.selectbox(
    "Choose a category for forecasting:",
    categories
)

category_df = df[
    df["Category"] == selected_category
].copy()


# ============================================================
# DAILY SALES FOR SELECTED CATEGORY
# ============================================================

daily_sales = (
    category_df
    .groupby("Date")["Units Sold"]
    .sum()
    .reset_index()
    .sort_values("Date")
)

daily_sales = daily_sales.set_index("Date")

# Fill missing dates with zero sales
daily_sales = daily_sales.asfreq("D")

daily_sales["Units Sold"] = (
    daily_sales["Units Sold"]
    .fillna(0)
)

daily_sales = daily_sales.reset_index()


st.subheader(
    f"📊 Daily Sales — {selected_category}"
)

st.dataframe(
    daily_sales.head(10),
    use_container_width=True
)


# ============================================================
# DAILY SALES TREND
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Daily Sales"
)

ax.set_title(
    f"Daily Sales Trend — {selected_category}"
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
# TRAIN / TEST SPLIT
# ============================================================

st.subheader("🔀 Chronological Train-Test Split")

split_index = int(
    len(daily_sales) * 0.8
)

train = daily_sales.iloc[:split_index].copy()
test = daily_sales.iloc[split_index:].copy()

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


fig, ax = plt.subplots(figsize=(14, 5))

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

ax.set_title(
    "Chronological Train-Test Split"
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
The data is split chronologically so that earlier observations are
used for training and later observations are reserved for testing.
This preserves the temporal structure of the dataset.
""")


# ============================================================
# MOVING AVERAGE
# ============================================================

st.subheader("📊 Moving Average Forecast")

window = 7

train["Moving Average"] = (
    train["Units Sold"]
    .rolling(window=window)
    .mean()
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    train["Date"],
    train["Units Sold"],
    label="Actual Sales"
)

ax.plot(
    train["Date"],
    train["Moving Average"],
    label="7-Day Moving Average"
)

ax.set_title(
    "7-Day Moving Average"
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
# MOVING AVERAGE TEST FORECAST
# ============================================================

moving_average_value = (
    train["Units Sold"]
    .tail(window)
    .mean()
)

test["Moving Average Forecast"] = (
    moving_average_value
)


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    test["Date"],
    test["Units Sold"],
    label="Actual Sales"
)

ax.plot(
    test["Date"],
    test["Moving Average Forecast"],
    label="Moving Average Forecast"
)

ax.set_title(
    "Moving Average Forecast vs Actual"
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
# MOVING AVERAGE METRICS
# ============================================================

mae = mean_absolute_error(
    test["Units Sold"],
    test["Moving Average Forecast"]
)

rmse = np.sqrt(
    mean_squared_error(
        test["Units Sold"],
        test["Moving Average Forecast"]
    )
)

mape = mean_absolute_percentage_error(
    test["Units Sold"],
    test["Moving Average Forecast"]
)


st.subheader("Moving Average Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

with col3:
    st.metric(
        "MAPE",
        f"{mape:.2%}"
    )


# ============================================================
# ARIMA
# ============================================================

st.subheader("📈 ARIMA Forecasting")

try:

    arima_model = ARIMA(
        train["Units Sold"],
        order=(1, 1, 1)
    )

    arima_result = arima_model.fit()

    arima_forecast = (
        arima_result
        .forecast(steps=len(test))
    )

    test["ARIMA Forecast"] = (
        arima_forecast.values
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        test["Date"],
        test["Units Sold"],
        label="Actual Sales"
    )

    ax.plot(
        test["Date"],
        test["ARIMA Forecast"],
        label="ARIMA Forecast"
    )

    ax.set_title(
        "ARIMA Forecast vs Actual"
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


    # ========================================================
    # ARIMA METRICS
    # ========================================================

    arima_mae = mean_absolute_error(
        test["Units Sold"],
        test["ARIMA Forecast"]
    )

    arima_rmse = np.sqrt(
        mean_squared_error(
            test["Units Sold"],
            test["ARIMA Forecast"]
        )
    )

    arima_mape = mean_absolute_percentage_error(
        test["Units Sold"],
        test["ARIMA Forecast"]
    )


    st.subheader("ARIMA Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "MAE",
            f"{arima_mae:.2f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{arima_rmse:.2f}"
        )

    with col3:
        st.metric(
            "MAPE",
            f"{arima_mape:.2%}"
        )


    arima_available = True

except Exception as e:

    st.warning(
        "ARIMA could not be fitted for this category. "
        "The Moving Average model will still be available."
    )

    arima_available = False


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("⚖️ Model Comparison")

if arima_available:

    comparison = pd.DataFrame({
        "Model": [
            "Moving Average",
            "ARIMA"
        ],
        "MAE": [
            mae,
            arima_mae
        ],
        "RMSE": [
            rmse,
            arima_rmse
        ],
        "MAPE": [
            mape,
            arima_mape
        ]
    })

else:

    comparison = pd.DataFrame({
        "Model": [
            "Moving Average"
        ],
        "MAE": [
            mae
        ],
        "RMSE": [
            rmse
        ],
        "MAPE": [
            mape
        ]
    })


st.dataframe(
    comparison,
    use_container_width=True
)


best_model = comparison.loc[
    comparison["MAPE"].idxmin(),
    "Model"
]

st.success(
    f"Best-performing model based on MAPE: **{best_model}**"
)


# ============================================================
# 90-DAY FUTURE FORECAST
# ============================================================

st.divider()

st.subheader("🔮 Next 90 Days Forecast")

st.write(
    f"""
The selected category is **{selected_category}**.
The model is now retrained using the complete historical
dataset for this category and used to estimate demand for
the next 90 days.
"""
)


# ------------------------------------------------------------
# ARIMA 90-DAY FORECAST
# ------------------------------------------------------------

future_dates = pd.date_range(
    start=daily_sales["Date"].max()
    + pd.Timedelta(days=1),
    periods=90,
    freq="D"
)


if best_model == "ARIMA" and arima_available:

    final_model = ARIMA(
        daily_sales["Units Sold"],
        order=(1, 1, 1)
    )

    final_result = final_model.fit()

    future_forecast = (
        final_result
        .forecast(steps=90)
        .values
    )

else:

    # Recursive 7-day moving-average forecast
    history = list(
        daily_sales["Units Sold"]
        .tail(7)
        .values
    )

    future_forecast = []

    for _ in range(90):

        prediction = np.mean(
            history[-7:]
        )

        future_forecast.append(
            prediction
        )

        history.append(
            prediction
        )


future_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Units Sold": future_forecast
})


# ============================================================
# FORECAST TABLE
# ============================================================

st.dataframe(
    future_df.head(20),
    use_container_width=True
)


# ============================================================
# 90-DAY FORECAST CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 6)
)

# Historical data
ax.plot(
    daily_sales["Date"],
    daily_sales["Units Sold"],
    label="Historical Sales"
)

# Future forecast
ax.plot(
    future_df["Date"],
    future_df["Forecasted Units Sold"],
    label=f"90-Day Forecast ({best_model})"
)

ax.axvline(
    daily_sales["Date"].max(),
    linestyle="--",
    label="Forecast Start"
)

ax.set_title(
    f"90-Day Demand Forecast — {selected_category}"
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
# FORECAST SUMMARY
# ============================================================

total_forecast = (
    future_df["Forecasted Units Sold"]
    .sum()
)

average_forecast = (
    future_df["Forecasted Units Sold"]
    .mean()
)

peak_forecast = (
    future_df["Forecasted Units Sold"]
    .max()
)


st.subheader("📋 90-Day Forecast Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Forecasted Demand",
        f"{total_forecast:,.0f}"
    )

with col2:
    st.metric(
        "Average Daily Demand",
        f"{average_forecast:,.0f}"
    )

with col3:
    st.metric(
        "Peak Forecasted Demand",
        f"{peak_forecast:,.0f}"
    )


# ============================================================
# BUSINESS INTERPRETATION
# ============================================================

st.info("""
The forecast provides an estimate of future demand based on
historical sales behavior. Businesses can use this information
to support inventory replenishment, procurement planning,
warehouse allocation and supplier coordination.

Forecasts should be interpreted together with business factors
such as promotions, holidays, seasonality and unexpected
operational changes.
""")