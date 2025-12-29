import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Price Optimization Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("💰 Retail Price Optimization Model")
st.markdown("Interactive dashboard for optimizing product prices based on market conditions")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")
    
    # Model loading
    st.subheader("Model Setup")
    model_file = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
    
    if model_file:
        try:
            model = joblib.load(model_file) if model_file.name.endswith('.joblib') else pickle.load(model_file)
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            model = None
    else:
        st.info("📁 Upload your trained model to proceed")
        model = None

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Analytics"])

with tab1:
    st.header("Single Price Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        # Adjust these inputs based on your model's features
        demand = st.number_input("Product Demand", min_value=0, value=100)
        competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=50.0)
        inventory = st.number_input("Current Inventory", min_value=0, value=500)
        seasonality = st.slider("Seasonality Factor", 0.5, 1.5, 1.0)
    
    with col2:
        st.subheader("Additional Parameters")
        cost = st.number_input("Product Cost ($)", min_value=0.0, value=20.0)
        elasticity = st.slider("Price Elasticity", -2.0, -0.5, -1.0)
        days_to_expiry = st.number_input("Days to Expiry", min_value=0, value=30)
    
    if st.button("🔮 Generate Optimal Price", type="primary"):
        if model is None:
            st.error("❌ Please upload a model first")
        else:
            try:
                # Prepare features (adjust based on your model)
                features = np.array([[demand, competitor_price, inventory, 
                                     seasonality, cost, elasticity, days_to_expiry]])
                
                # Make prediction
                optimal_price = model.predict(features)[0]
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Price", f"${optimal_price:.2f}")
                with col2:
                    st.metric("Margin", f"${optimal_price - cost:.2f}", f"{((optimal_price-cost)/cost*100):.1f}%")
                with col3:
                    st.metric("Competitor Price", f"${competitor_price:.2f}")
                
                st.success(f"✅ Recommended selling price: **${optimal_price:.2f}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab2:
    st.header("Batch Price Optimization")
    st.markdown("Upload a CSV file to optimize prices for multiple products")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file and model:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("🚀 Optimize All Prices"):
                # Extract features and predict
                feature_cols = [col for col in df.columns if col not in ['product_id', 'product_name']]
                X = df[feature_cols].values
                predictions = model.predict(X)
                
                results = df.copy()
                results['optimized_price'] = predictions
                
                st.success("✅ Optimization complete!")
                st.dataframe(results)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Optimized Prices",
                    data=csv,
                    file_name=f"optimized_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    elif uploaded_file and not model:
        st.error("❌ Please upload a model first")

with tab3:
    st.header("Model Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        if model:
            st.json({
                "Model Type": type(model).__name__,
                "Status": "✅ Loaded",
                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            st.warning("⚠️ No model loaded")
    
    with col2:
        st.subheader("Performance Metrics")
        st.info("Upload metrics from your notebook evaluation here")
        metric_data = st.text_area("Paste your metrics (R², MAE, RMSE, etc.)", height=100)

# Footer
st.markdown("---")
st.markdown("**Deployment Ready** | For cloud deployment, use: `streamlit run app.py`")