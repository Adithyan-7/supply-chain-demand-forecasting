import streamlit as st

st.set_page_config(
    page_title="Business Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Business Insights & Recommendations")

st.markdown("""
This section summarizes the key findings from Exploratory Data Analysis,
Anomaly Detection, and Demand Forecasting and translates them into
business recommendations for inventory and supply chain management.
""")

st.divider()


# --------------------------------------------------
# EDA Insights
# --------------------------------------------------

st.subheader("📊 EDA Insights")

st.markdown("""
- Sales performance varies across products and categories.
- Daily and monthly sales trends reveal changes in demand over time.
- Inventory movement varies across products, highlighting opportunities
  for better inventory allocation.
- Some products experience considerably higher sales volumes than others.
- Store and regional analysis helps identify differences in demand across locations.
""")


# --------------------------------------------------
# Anomaly Detection Insights
# --------------------------------------------------

st.subheader("🚨 Anomaly Detection Insights")

st.markdown("""
- Z-Score and IQR methods identify unusually high or low sales observations.
- Positive anomalies may indicate promotions, seasonal demand, special events,
  or unusually large purchases.
- Negative anomalies may indicate stock shortages, operational issues,
  supplier delays, or reduced customer demand.
- Monitoring these anomalies allows unusual demand patterns to be investigated
  before they significantly affect inventory operations.
""")

st.info("""
Anomaly detection can help supply chain teams identify unexpected demand
fluctuations and investigate their potential causes.
""")


# --------------------------------------------------
# Demand Forecasting Insights
# --------------------------------------------------

st.subheader("📈 Demand Forecasting Insights")

st.markdown("""
- Historical sales data can be used to estimate future demand.
- Moving Average provides a simple baseline forecasting approach.
- ARIMA provides an additional time-series forecasting approach.
- MAE, RMSE, and MAPE allow forecasting performance to be evaluated
  quantitatively.
- Forecast information can support procurement, inventory planning,
  and supplier coordination.
""")

st.info("""
Lower forecasting errors indicate more reliable demand estimates and can
support better inventory planning decisions.
""")


# --------------------------------------------------
# Business Recommendations
# --------------------------------------------------

st.subheader("🎯 Business Recommendations")

st.markdown("""
1. **Prioritize high-demand products** when planning inventory replenishment.

2. **Monitor detected anomalies** and investigate repeated demand spikes
   or drops.

3. **Use demand forecasts for procurement planning** rather than relying
   only on historical averages.

4. **Review slow-moving inventory regularly** to reduce unnecessary
   storage and holding costs.

5. **Coordinate inventory planning with seasonal demand patterns** identified
   from historical sales.

6. **Monitor inventory and sales together** to identify potential stockout
   or overstock situations.

7. **Track forecasting accuracy continuously** using MAE, RMSE, and MAPE.

8. **Investigate store and regional differences** when allocating inventory.
""")


# --------------------------------------------------
# Limitations
# --------------------------------------------------

st.subheader("⚠️ Limitations")

st.markdown("""
- Forecast accuracy depends heavily on the quality and amount of historical data.
- External market conditions and competitor actions may affect future demand.
- Statistical anomaly detection can occasionally flag normal variations.
- Detected anomalies indicate unusual observations but do not automatically
  identify their root cause.
- Forecasting results should therefore be interpreted together with
  operational and business knowledge.
""")


# --------------------------------------------------
# Future Improvements
# --------------------------------------------------

st.subheader("🔧 Future Improvements")

st.markdown("""
- Incorporate external variables such as promotions, holidays and weather.
- Explore additional forecasting approaches and compare their performance.
- Implement automated or real-time anomaly monitoring.
- Add product/category-level interactive forecasting.
- Develop inventory optimization recommendations based on forecasted demand.
""")


# --------------------------------------------------
# Final Conclusion
# --------------------------------------------------

st.divider()

st.subheader("Conclusion")

st.success("""
Combining exploratory analysis, anomaly detection and demand forecasting
provides a more complete view of supply chain demand.

Historical patterns explain past behavior, anomaly detection highlights
unusual events, and forecasting provides estimates of future demand.
Together, these analyses can support better inventory planning and
data-driven supply chain decisions.
""")