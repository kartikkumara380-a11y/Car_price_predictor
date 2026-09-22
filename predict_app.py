import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

model = joblib.load('models/car_price_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

# Extract available brands from model columns
brand_cols = [c for c in model_columns if c.startswith('Brand_')]
brand_names = sorted([c.replace('Brand_', '') for c in brand_cols])

st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 0;
    }
    .hero p {
        color: #999;
        font-size: 1.1rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .price-card h2 {
        color: #4ADE80;
        font-size: 2.5rem;
        margin: 0.5rem 0;
    }
    .price-range {
        color: #999;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <h1>🚗 Car Price Predictor</h1>
        <p>Get an instant, data-driven resale estimate for your car</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Input form
col1, col2, col3 = st.columns(3)
with col1:
    brand = st.selectbox("🚘 Car Brand/Model", brand_names)
    present_price = st.number_input("💰 Present Price (in lakhs)", min_value=0.0, value=5.0, step=0.5)
with col2:
    car_age = st.slider("📅 Car Age (years)", 0, 20, 5)
    kms_driven = st.number_input("🛣️ Kms Driven", min_value=0, value=30000, step=1000)
with col3:
    fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

col4, col5 = st.columns(2)
with col4:
    seller_type = st.selectbox("🏪 Seller Type", ["Dealer", "Individual"])
with col5:
    owner = st.selectbox("👤 Previous Owners", [0, 1, 2, 3])

predict_clicked = st.button("🔍 Check Value")

st.divider()

if predict_clicked:
    input_data = dict.fromkeys(model_columns, 0)
    input_data['Present_Price'] = present_price
    input_data['Kms_Driven'] = kms_driven
    input_data['Car_Age'] = car_age
    input_data['Owner'] = owner
    input_data['Price_per_Age'] = present_price / (car_age + 1)
    input_data['Price_per_Km'] = present_price / (kms_driven + 1)

    brand_col = f'Brand_{brand}'
    if brand_col in input_data:
        input_data[brand_col] = 1

    if fuel_type == 'Diesel':
        input_data['Fuel_Type_Diesel'] = 1
    elif fuel_type == 'Petrol':
        input_data['Fuel_Type_Petrol'] = 1
    if seller_type == 'Individual':
        input_data['Seller_Type_Individual'] = 1
    if transmission == 'Manual':
        input_data['Transmission_Manual'] = 1

    input_df = pd.DataFrame([input_data])[model_columns]
    prediction = model.predict(input_df)[0]

    low = prediction * 0.92
    high = prediction * 1.08

    result_col, info_col = st.columns([1, 1])
    with result_col:
        st.markdown(f"""
            <div class="price-card">
                <div style="color:#999;">Estimated Resale Value</div>
                <h2>₹{prediction:.2f} Lakhs</h2>
                <div class="price-range">Likely range: ₹{low:.2f}L - ₹{high:.2f}L</div>
            </div>
        """, unsafe_allow_html=True)

    with info_col:
        st.markdown("#### How this is calculated")
        st.write(f"Based on a machine learning model trained on real used car sales data, achieving **~86% accuracy (R² score)**.")
        depreciation = ((present_price - prediction) / present_price * 100) if present_price > 0 else 0
        st.metric("Depreciation from original price", f"{depreciation:.1f}%")

    chart_data = pd.DataFrame({
        "Price Type": ["Original Price", "Estimated Resale"],
        "Amount (Lakhs)": [present_price, prediction]
    })
    st.bar_chart(chart_data.set_index("Price Type"))

    st.caption("⚠️ This is an estimate based on historical data and may not reflect current market conditions, vehicle condition, or location-specific pricing.")
else:
    st.info("Fill in your car's details above and click **Check Value** to get an instant estimate.")