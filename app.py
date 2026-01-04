import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None

# Sidebar
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")
    
    # Model loading
    st.subheader("Model Setup")
    
    # Option 1: Upload model file
    model_file = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
    
    # Option 2: Provide model parameters for a demo model
    st.markdown("---")
    st.subheader("Demo Mode")
    use_demo = st.checkbox("Use Demo Model", value=False)
    
    if use_demo:
        # Create a simple demo model
        from sklearn.ensemble import RandomForestRegressor
        
        # Create and train a simple demo model
        np.random.seed(42)
        n_samples = 1000
        X_demo = np.random.randn(n_samples, 7)  # 7 features
        y_demo = 20 + X_demo[:, 1] * 15 + np.random.randn(n_samples) * 5  # Price
        
        demo_model = RandomForestRegressor(n_estimators=10, random_state=42)
        demo_model.fit(X_demo, y_demo)
        
        st.session_state.model = demo_model
        st.session_state.model_info = {
            "name": "Demo Random Forest",
            "n_features": 7,
            "features": ["demand", "competitor_price", "inventory", 
                        "seasonality", "cost", "elasticity", "days_to_expiry"],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.success("✅ Demo model loaded successfully")
    
    # Load model from uploaded file
    elif model_file:
        try:
            # Read file bytes
            file_bytes = model_file.read()
            
            # Try joblib first
            try:
                import io
                st.session_state.model = joblib.load(io.BytesIO(file_bytes))
                st.success("✅ Model loaded with joblib")
            except:
                # Try pickle
                try:
                    st.session_state.model = pickle.loads(file_bytes)
                    st.success("✅ Model loaded with pickle")
                except Exception as e:
                    st.error(f"Pickle error: {str(e)}")
                    st.session_state.model = None
            
            if st.session_state.model:
                st.session_state.model_info = {
                    "name": type(st.session_state.model).__name__,
                    "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": model_file.name
                }
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("💡 Tip: Try saving your model with joblib for better compatibility")
    else:
        if not use_demo:
            st.info("📁 Upload your trained model or enable Demo Mode to proceed")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Prediction", "Model Info", "Demo Data"])

with tab1:
    st.header("Single Price Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        
        # Dynamically create inputs based on model info
        if st.session_state.model_info and 'features' in st.session_state.model_info:
            features = st.session_state.model_info['features']
            feature_values = {}
            
            for i, feature in enumerate(features):
                if feature == "demand":
                    feature_values[feature] = st.number_input("Product Demand", min_value=0, value=100, key=f"demand_{i}")
                elif feature == "competitor_price":
                    feature_values[feature] = st.number_input("Competitor Price ($)", min_value=0.0, value=50.0, key=f"comp_price_{i}")
                elif feature == "inventory":
                    feature_values[feature] = st.number_input("Current Inventory", min_value=0, value=500, key=f"inv_{i}")
                elif feature == "seasonality":
                    feature_values[feature] = st.slider("Seasonality Factor", 0.5, 1.5, 1.0, key=f"season_{i}")
                elif feature == "cost":
                    feature_values[feature] = st.number_input("Product Cost ($)", min_value=0.0, value=20.0, key=f"cost_{i}")
                elif feature == "elasticity":
                    feature_values[feature] = st.slider("Price Elasticity", -2.0, -0.5, -1.0, key=f"elasticity_{i}")
                elif feature == "days_to_expiry":
                    feature_values[feature] = st.number_input("Days to Expiry", min_value=0, value=30, key=f"expiry_{i}")
                else:
                    feature_values[feature] = st.number_input(f"{feature}", value=0.0, key=f"feature_{i}")
        else:
            # Default features if no model info
            demand = st.number_input("Product Demand", min_value=0, value=100)
            competitor_price = st.number_input("Competitor Price ($)", min_value=0.0, value=50.0)
            inventory = st.number_input("Current Inventory", min_value=0, value=500)
            seasonality = st.slider("Seasonality Factor", 0.5, 1.5, 1.0)
            cost = st.number_input("Product Cost ($)", min_value=0.0, value=20.0)
            elasticity = st.slider("Price Elasticity", -2.0, -0.5, -1.0)
            days_to_expiry = st.number_input("Days to Expiry", min_value=0, value=30)
            
            feature_values = {
                "demand": demand,
                "competitor_price": competitor_price,
                "inventory": inventory,
                "seasonality": seasonality,
                "cost": cost,
                "elasticity": elasticity,
                "days_to_expiry": days_to_expiry
            }
    
    with col2:
        st.subheader("Price Settings")
        min_price = st.number_input("Minimum Price ($)", min_value=0.0, value=cost if 'cost' in locals() else 20.0)
        max_price = st.number_input("Maximum Price ($)", min_value=0.0, value=100.0)
        
        st.markdown("---")
        st.subheader("Prediction Controls")
        confidence_level = st.slider("Confidence Level", 0.5, 0.99, 0.95)
    
    if st.button("🔮 Generate Optimal Price", type="primary"):
        if st.session_state.model is None:
            st.error("❌ Please upload a model or enable Demo Mode first")
        else:
            try:
                # Prepare features array
                if st.session_state.model_info and 'features' in st.session_state.model_info:
                    features_list = st.session_state.model_info['features']
                    X = np.array([[feature_values.get(f, 0) for f in features_list]])
                else:
                    X = np.array([[demand, competitor_price, inventory, seasonality, 
                                 cost, elasticity, days_to_expiry]])
                
                # Make prediction
                optimal_price = st.session_state.model.predict(X)[0]
                
                # Apply price bounds
                optimal_price = max(min_price, min(max_price, optimal_price))
                
                # Display results
                st.subheader("📊 Optimization Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Optimal Price", f"${optimal_price:.2f}")
                with col2:
                    margin = optimal_price - cost
                    margin_pct = (margin / cost * 100) if cost > 0 else 0
                    st.metric("Margin", f"${margin:.2f}", f"{margin_pct:.1f}%")
                with col3:
                    st.metric("Competitor Price", f"${competitor_price:.2f}")
                with col4:
                    price_diff = optimal_price - competitor_price
                    st.metric("Price Difference", f"${price_diff:.2f}")
                
                # Visualization
                st.markdown("---")
                st.subheader("Price Comparison")
                
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[
                    go.Bar(
                        name='Prices',
                        x=['Cost', 'Competitor', 'Optimal'],
                        y=[cost, competitor_price, optimal_price],
                        text=[f'${cost:.2f}', f'${competitor_price:.2f}', f'${optimal_price:.2f}'],
                        textposition='auto',
                        marker_color=['red', 'orange', 'green']
                    )
                ])
                
                fig.update_layout(
                    title="Price Comparison",
                    yaxis_title="Price ($)",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Check if your model expects the same features you're providing")

with tab2:
    st.header("Batch Price Optimization")
    st.markdown("Upload a CSV file to optimize prices for multiple products")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv", key="batch_csv")
    
    if uploaded_file and st.session_state.model:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Show column information
            st.subheader("Data Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Missing Values': df.isnull().sum().values,
                'Unique Values': df.nunique().values
            })
            st.dataframe(col_info)
            
            if st.button("🚀 Optimize All Prices", type="primary"):
                with st.spinner("Optimizing prices..."):
                    # Extract features
                    # Assuming the CSV has the same feature columns
                    if st.session_state.model_info and 'features' in st.session_state.model_info:
                        feature_cols = st.session_state.model_info['features']
                    else:
                        feature_cols = [col for col in df.columns if col not in ['product_id', 'product_name', 'price']]
                    
                    # Check if features exist in dataframe
                    missing_features = [f for f in feature_cols if f not in df.columns]
                    if missing_features:
                        st.error(f"Missing features in CSV: {missing_features}")
                        st.stop()
                    
                    X = df[feature_cols].values
                    predictions = st.session_state.model.predict(X)
                    
                    results = df.copy()
                    results['optimized_price'] = predictions.round(2)
                    results['margin'] = (results['optimized_price'] - results.get('cost', 0)).round(2)
                    
                    st.success(f"✅ Optimized {len(results)} products!")
                    
                    # Display results
                    st.subheader("Optimized Prices")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Optimized Prices",
                        data=csv,
                        file_name=f"optimized_prices_{timestamp}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV has the correct feature columns")
    elif uploaded_file and not st.session_state.model:
        st.error("❌ Please upload a model first or enable Demo Mode")

with tab3:
    st.header("Model Information")
    
    if st.session_state.model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            model_info = st.session_state.model_info or {}
            
            info_dict = {
                "Model Type": type(st.session_state.model).__name__,
                "Status": "✅ Loaded",
                "Features": model_info.get('features', 'Unknown'),
                "Number of Features": model_info.get('n_features', 'Unknown'),
                "Loaded At": model_info.get('uploaded', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            for key, value in info_dict.items():
                st.write(f"**{key}:** {value}")
            
            # Try to get model parameters
            if hasattr(st.session_state.model, 'get_params'):
                st.subheader("Model Parameters")
                params = st.session_state.model.get_params()
                st.json(params)
        
        with col2:
            st.subheader("Model Performance")
            
            # Placeholder for metrics
            st.info("Model metrics would appear here after training")
            
            # Upload metrics
            uploaded_metrics = st.file_uploader("Upload metrics JSON", type="json")
            if uploaded_metrics:
                try:
                    metrics = json.load(uploaded_metrics)
                    st.json(metrics)
                    
                    # Display as metrics
                    cols = st.columns(3)
                    metric_items = list(metrics.items())
                    
                    for i, (key, value) in enumerate(metric_items[:3]):
                        with cols[i]:
                            st.metric(key, f"{value:.4f}" if isinstance(value, (int, float)) else value)
                    
                    # Add more if needed
                    if len(metric_items) > 3:
                        cols = st.columns(3)
                        for i, (key, value) in enumerate(metric_items[3:6], 3):
                            with cols[i-3]:
                                st.metric(key, f"{value:.4f}" if isinstance(value, (int, float)) else value)
                except:
                    st.error("Could not load metrics file")
    
    else:
        st.warning("⚠️ No model loaded. Upload a model or enable Demo Mode.")

with tab4:
    st.header("Demo Data & Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quick Start Guide")
        st.markdown("""
        1. **Enable Demo Mode** in the sidebar to test the dashboard
        2. **Single Prediction**: Adjust sliders and see optimal prices
        3. **Batch Prediction**: Use the sample CSV below
        4. **Upload Your Model**: When ready, upload your trained model
        
        ### Sample Data Format
        Your CSV should include these columns:
        - `product_id` (optional): Unique identifier
        - `product_name` (optional): Product description
        - `demand`: Expected demand units
        - `competitor_price`: Competitor's price in $
        - `inventory`: Current stock level
        - `seasonality`: Seasonal factor (0.5-1.5)
        - `cost`: Product cost in $
        - `elasticity`: Price elasticity (-2.0 to -0.5)
        - `days_to_expiry`: Days until expiry
        """)
    
    with col2:
        st.subheader("Generate Sample Data")
        
        n_samples = st.number_input("Number of sample products", min_value=5, max_value=100, value=10)
        
        if st.button("Generate Sample CSV"):
            # Create sample data
            np.random.seed(42)
            
            sample_data = {
                'product_id': [f'PROD_{i:03d}' for i in range(1, n_samples+1)],
                'product_name': [f'Product {i}' for i in range(1, n_samples+1)],
                'demand': np.random.randint(50, 500, n_samples),
                'competitor_price': np.random.uniform(30, 80, n_samples).round(2),
                'inventory': np.random.randint(100, 1000, n_samples),
                'seasonality': np.random.uniform(0.5, 1.5, n_samples).round(2),
                'cost': np.random.uniform(15, 40, n_samples).round(2),
                'elasticity': np.random.uniform(-2.0, -0.5, n_samples).round(2),
                'days_to_expiry': np.random.randint(1, 60, n_samples)
            }
            
            df_sample = pd.DataFrame(sample_data)
            st.dataframe(df_sample)
            
            # Download sample
            csv = df_sample.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_products.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
**Deployment Ready** | For cloud deployment, use: `streamlit run app.py`
- **Streamlit Cloud**: https://share.streamlit.io
- **Docker**: Use official Streamlit Docker image
- **Local**: Run `pip install -r requirements.txt` then `streamlit run app.py`
""")

# Add requirements
with st.sidebar:
    st.markdown("---")
    with st.expander("📋 Requirements"):
        st.code("""
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
plotly==5.17.0
        """)