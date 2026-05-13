import streamlit as st

st.title("Late Delivery Risk Prediction")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# 2. PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="🚚",
    layout="wide"
)



# =========================
# 4. LOAD DATASET
# =========================

df = pd.read_csv(
    "data/featured_apl_logistics.csv"
)

# =========================
# 5. TITLE
# =========================

st.title("🚚 Late Delivery Risk Prediction Dashboard")

st.markdown("""
This system predicts the probability of shipment delays using Machine Learning.
""")

# =========================
# 6. SIDEBAR
# =========================

st.sidebar.header("Filter Options")

shipping_mode = st.sidebar.selectbox(
    "Select Shipping Mode",
    ["Standard Class", "Second Class", "First Class", "Same Day"]
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["Consumer", "Corporate", "Home Office"]
)

risk_threshold = st.sidebar.slider(
    "Risk Threshold",
    0.0,
    1.0,
    0.7
)

# =========================
# 7. KPI SECTION
# =========================

total_orders = len(df)

high_risk_orders = len(
    df[df['Late_delivery_risk'] == 1]
)

avg_sales = round(df['Sales'].mean(), 2)

avg_profit = round(
    df['Order_Profit_Per_Order'].mean(),
    2
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Orders",
    total_orders
)

col2.metric(
    "High Risk Orders",
    high_risk_orders
)

col3.metric(
    "Average Sales",
    f"${avg_sales}"
)

col4.metric(
    "Average Profit",
    f"${avg_profit}"
)

# =========================
# 8. RISK DISTRIBUTION
# =========================

st.subheader("Late Delivery Risk Distribution")

fig = px.histogram(
    df,
    x='Late_delivery_risk',
    color='Late_delivery_risk',
    title='Late Delivery Risk Distribution'
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================
# 9. SHIPPING MODE ANALYSIS
# =========================

st.subheader("Shipping Mode Analysis")

shipping_analysis = (
    df.groupby('Shipping_Mode_Standard Class')
    ['Late_delivery_risk']
    .mean()
    .reset_index()
)

fig2 = px.bar(
    shipping_analysis,
    x='Shipping_Mode_Standard Class',
    y='Late_delivery_risk',
    title='Shipping Mode vs Delay Risk'
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================
# 10. ORDER RISK PREDICTION
# =========================

st.subheader("Predict Shipment Delay Risk")

col1, col2 = st.columns(2)

with col1:

    quantity = st.number_input(
        "Order Quantity",
        min_value=1,
        max_value=100,
        value=5
    )

    sales = st.number_input(
        "Sales Amount",
        min_value=0.0,
        value=500.0
    )

    product_price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=100.0
    )

with col2:

    shipping_days = st.slider(
        "Scheduled Shipping Days",
        1,
        10,
        3
    )

    discount_rate = st.slider(
        "Discount Rate",
        0.0,
        1.0,
        0.1
    )

    profit = st.number_input(
        "Order Profit",
        value=50.0
    )

# =========================
# 11. PREDICTION BUTTON
# =========================

if st.button("Predict Late Delivery Risk"):

    # Example feature engineering

    shipping_pressure = quantity / (shipping_days + 1)

    order_complexity = quantity * product_price

    # Create sample input dataframe

    sample = pd.DataFrame({

        'Order_Item_Quantity': [quantity],

        'Sales': [sales],

        'Order_Item_Product_Price': [product_price],

        'Days_for_shipment_(scheduled)': [shipping_days],

        'Order_Item_Discount_Rate': [discount_rate],

        'Order_Profit_Per_Order': [profit],

        'Shipping_Pressure_Index': [shipping_pressure],

        'Order_Complexity_Score': [order_complexity]

    })

    # Add missing columns with 0
    missing_cols = set(model.feature_names_in_) - set(sample.columns)

    for col in missing_cols:
        sample[col] = 0

    # Arrange columns properly
    sample = sample[model.feature_names_in_]

    # Prediction
    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Results")

    st.write(
        f"Late Delivery Probability: {probability:.2f}"
    )

    # Risk category
    if probability >= 0.7:

        st.error("⚠️ High Risk Shipment")

    elif probability >= 0.3:

        st.warning("⚠️ Medium Risk Shipment")

    else:

        st.success("✅ Low Risk Shipment")

# =========================
# 12. TOP HIGH RISK ORDERS
# =========================

st.subheader("High Risk Orders")

high_risk_df = df[
    df['Late_delivery_risk'] == 1
].head(10)

st.dataframe(high_risk_df)

