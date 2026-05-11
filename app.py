import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. UI Configuration ---
st.set_page_config(
    page_title="Poverty Prediction AI System",
    page_icon="🏠",
    layout="wide"
)

# Title and Description
st.title("🏠 Costa Rica Household Poverty Prediction")
st.markdown("""
### Welcome to the Poverty Prediction System
This system utilizes an optimized **XGBoost** algorithm to analyze socio-economic household data and classify them:
* **Poverty (1):** Households eligible for assistance.
* **Non-Poverty (0):** Self-sufficient households.
""")

# --- 2. Model Loading ---
@st.cache_resource
def load_trained_model():
    # Ensure the file name matches the one saved from your notebook
    return joblib.load('xgboost_poverty_model.pkl')

try:
    model = load_trained_model()
    st.sidebar.success("✅ XGBoost Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.error("Make sure 'xgboost_poverty_model.pkl' exists in the same directory.")
    st.stop()

# --- 3. Sidebar & Upload ---
st.sidebar.header("Test Area")
st.sidebar.info("Upload a CSV file containing household data for the AI to classify.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# --- 4. Processing & Prediction ---
if uploaded_file is not None:
    # Read the uploaded file
    test_df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Uploaded Data Preview")
    st.write(f"Found **{test_df.shape[0]}** households for analysis.")
    st.dataframe(test_df.head(10))

    if st.button("🚀 Start Smart Prediction"):
        with st.spinner('Analyzing data and processing features...'):
            try:
                # Create a copy for processing
                X_processed = test_df.copy()

                # --- Preprocessing Step to solve ValueError ---
                # 1. Drop unnecessary columns
                cols_to_drop = ['Id', 'idhogar', 'Target', 'Target_Binary']
                for col in cols_to_drop:
                    if col in X_processed.columns:
                        X_processed = X_processed.drop(columns=[col])

                # 2. Convert 'yes' and 'no' to numerical values
                mapping = {'yes': 1, 'no': 0}
                object_cols = ['dependency', 'edjefe', 'edjefa']
                for col in object_cols:
                    if col in X_processed.columns:
                        X_processed[col] = X_processed[col].replace(mapping).astype(float)

                # --- Execution ---
                preds = model.predict(X_processed)
                
                # Append results to the original dataframe
                test_df['Prediction'] = preds
                test_df['Status'] = test_df['Prediction'].map({1: 'Poverty 🔴', 0: 'Non-Poverty 🟢'})

                st.success("✨ Analysis Completed Successfully!")

                # --- 5. Results Visualization ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎯 Detailed Classification Results")
                    # Display basic columns along with the status
                    display_cols = ['Id'] if 'Id' in test_df.columns else []
                    display_cols += ['Status']
                    st.dataframe(test_df[display_cols + list(X_processed.columns[:3])].head(50))

                with col2:
                    st.subheader("📊 Batch Statistics")
                    fig, ax = plt.subplots()
                    sns.countplot(data=test_df, x='Status', palette={'Poverty 🔴': '#E63946', 'Non-Poverty 🟢': '#1D9E75'})
                    plt.title("Poverty Distribution in Uploaded Sample")
                    st.pyplot(fig)

                # Download Results Button
                csv_output = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv_output,
                    file_name='poverty_predictions_output.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Ensure the CSV file contains the same features the model was trained on.")
else:
    # Welcome message when no file is uploaded
    st.warning("👈 Please upload a data file from the sidebar to begin.")
    
    st.markdown("""
    ---
    ### 💡 How does this work?
    1.  **Upload:** Upload a CSV file containing Costa Rican household data.
    2.  **Process:** The system automatically cleans the data (e.g., converting 'yes/no' to numbers and removing identifiers).
    3.  **Predict:** The model provides the final result for each household instantly, with an option to download a complete report.
    """)