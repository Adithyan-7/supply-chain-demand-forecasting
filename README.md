# Supply Chain Analytics — Demand Forecasting & Anomaly Detection

A data analytics project focused on retail supply-chain demand, combining data preprocessing, exploratory data analysis, statistical anomaly detection, demand forecasting, and an interactive Streamlit dashboard.

## Project Overview

The project analyzes retail store inventory and sales data to understand demand patterns, identify unusual sales behavior, forecast future demand, and provide business recommendations for inventory and supply-chain planning.

**Workflow:** Data Preprocessing → Exploratory Data Analysis → Anomaly Detection → Demand Forecasting → Business Insights

## Objectives

- Clean and prepare retail inventory and sales data.
- Analyze sales, inventory, category, and regional patterns.
- Identify unusual demand patterns using statistical methods.
- Compare Moving Average and ARIMA forecasting approaches.
- Forecast demand for the next 90 days for a selected product category.
- Present analytical results through an interactive Streamlit dashboard.
- Generate business recommendations supporting inventory planning and decision-making.

## Dataset

The dataset contains retail inventory and sales observations with fields including:

`Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Units Ordered`, `Demand Forecast`, `Price`, `Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, and `Seasonality`.

The `Date` field is used as the time component for the daily sales time series.

## Project Structure

```text
supply-chain-demand-forecasting/
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── 1_EDA.py
│       ├── 2_Anomaly_Detection.py
│       ├── 3_Demand_Forecasting.py
│       └── 4_Business_Insights.py
│
├── data/
│   ├── raw/
│   │   └── retail_store_inventory.csv
│   ├── cleaned/
│   │   └── retail_store_inventory_cleaned.csv
│   └── processed/
│       └── daily_sales.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_anomaly_detection.ipynb
│   └── 04_demand_forecasting.ipynb
│
├── reports/
├── visuals/
├── requirements.txt
└── .gitignore
```

## 1. Data Preprocessing

The preprocessing notebook prepares the raw retail dataset for analysis. It includes dataset inspection, missing-value and duplicate checks, numerical validation, date conversion, chronological sorting, and daily aggregation of `Units Sold`.

The daily time series is checked for missing dates. A complete daily date range is created and missing daily sales values are handled using linear interpolation where required. Cleaned and processed datasets are saved for later stages.

### Time-Series Decomposition

The prepared daily sales series is decomposed using `statsmodels` into:

- Observed component
- Trend
- Seasonal component
- Residual component

A 7-day period is used to examine weekly seasonality in the daily retail sales series.

## 2. Exploratory Data Analysis

The EDA notebook and Streamlit page explore the main characteristics of the retail dataset, including:

- Sales patterns
- Inventory levels
- Product categories
- Regional performance
- Daily and monthly sales behavior
- Product/category-level comparisons
- Relationships between important numerical variables

EDA provides the foundation for the anomaly detection, forecasting, and business-insight stages.

## 3. Anomaly Detection

The project identifies unusual daily sales observations using two statistical approaches.

### Z-Score

The Z-Score method identifies observations that deviate significantly from the mean. A threshold of ±3 is used in the dashboard to flag potential anomalies.

### IQR

The Interquartile Range method identifies observations outside:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

### Dashboard Features

The Streamlit anomaly-detection page provides:

- Product-category selection
- Category-specific daily sales
- Z-Score anomaly counts
- IQR anomaly counts
- Historical anomaly visualizations
- Comparison of Z-Score and IQR results
- Detected anomaly tables
- Business interpretation

An anomaly is treated as an observation requiring investigation rather than automatically being classified as an error.

## 4. Demand Forecasting

Two forecasting approaches are evaluated:

### 7-Day Moving Average

The Moving Average model provides a simple baseline by using recent sales values to estimate demand.

### ARIMA

An ARIMA model is used to model the historical sales time series and generate forecasts.

Models are evaluated using:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **MAPE — Mean Absolute Percentage Error**

Lower error values indicate better forecasting performance.

### 90-Day Forecast

The Streamlit dashboard allows the user to select a product category and generates a forecast for the next **90 days** using the best-performing available forecasting approach based on MAPE.

The dashboard displays historical sales, future forecast, forecast start point, forecasted daily demand, total forecasted demand, average daily demand, and peak forecasted demand.

## 5. Business Insights

The business-insights module translates analytical findings into operational recommendations.

Recommendation areas include:

- Maintaining appropriate inventory levels for high-demand products.
- Investigating recurring sales anomalies.
- Using forecasts for procurement and supplier planning.
- Monitoring unusual demand patterns.
- Optimizing replenishment schedules.
- Preparing for seasonal demand.
- Reviewing slow-moving inventory.
- Improving data-quality monitoring.
- Integrating forecasting results into planning workflows.

Forecasts and anomaly results should be interpreted alongside operational and business context.

## Streamlit Dashboard

The application contains four analytical modules:

### Exploratory Analysis
Interactive exploration of sales, inventory, category, and regional patterns.

### Anomaly Detection
Category selection and historical anomaly investigation using Z-Score and IQR methods.

### Demand Forecasting
Category selection, model comparison, forecasting evaluation, and next-90-day demand forecasting.

### Business Insights
Business findings, recommendations, limitations, and future improvements.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Statsmodels
- Streamlit
- Jupyter Notebook
- Git / GitHub

## Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd supply-chain-demand-forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

From the project root:

```bash
streamlit run app/app.py
```

## Limitations

- Forecast performance depends on data quality and historical coverage.
- External factors such as market changes, competitor actions, and economic conditions are not fully represented.
- Statistical anomaly methods can flag legitimate but unusual business events.
- Forecasts should not be treated as guaranteed future demand.
- Business decisions should combine model outputs with operational knowledge.

## Future Improvements

- Test additional forecasting models such as Prophet or XGBoost.
- Incorporate promotions, holidays, weather, and other external variables.
- Develop real-time anomaly monitoring.
- Add automated inventory optimization.
- Compare additional forecasting techniques.
- Improve automated business recommendations.

## Conclusion

This project combines data preprocessing, exploratory analysis, statistical anomaly detection, time-series forecasting, and business insights into an interactive supply-chain analytics application.

The dashboard provides category-level analysis, historical anomaly identification, model evaluation, and 90-day demand forecasts to support more informed inventory and supply-chain planning.
