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