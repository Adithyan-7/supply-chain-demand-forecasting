import streamlit as st

st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain Analytics Dashboard")

st.markdown("""
Welcome to the **Supply Chain Analytics Dashboard**.

This project analyzes retail inventory data to:

- 📊 Explore inventory and sales trends
- 🚨 Detect unusual demand patterns
- 📈 Forecast future product demand
- 💡 Support inventory planning and supply chain decisions

Use the navigation menu on the left to explore the different sections of the project.
""")

st.divider()

st.subheader("Project Objectives")

col1, col2 = st.columns(2)

with col1:
    st.info("Analyze retail inventory data")
    st.info("Identify anomalies in demand")

with col2:
    st.success("Forecast future sales")
    st.success("Support business decision-making")

st.divider()

st.subheader("Technologies Used")

st.markdown("""
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels (ARIMA)
- Streamlit
""")

st.divider()

st.subheader("Project Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### 📊 Exploratory Analysis
        
        Understand sales, inventory,
        category and regional patterns.
        """
    )

with col2:
    st.warning(
        """
        ### 🚨 Anomaly Detection
        
        Identify unusual demand patterns
        using Z-Score and IQR methods.
        """
    )

with col3:
    st.success(
        """
        ### 📈 Demand Forecasting
        
        Forecast future demand using
        Moving Average and ARIMA.
        """
    )

    st.divider()

st.subheader("Project Workflow")

st.markdown("""
**Data Preprocessing → Exploratory Analysis → Anomaly Detection → Demand Forecasting → Business Insights**
""")

st.divider()

st.caption(
    "Supply Chain Analytics — Demand Forecasting & Anomaly Detection"
)