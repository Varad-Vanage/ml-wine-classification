import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("🍷 Red Wine Quality Classification Dashboard")
st.write("Upload test data, pick a model, and evaluate performance metrics in real time.")

# Sidebar - Dataset Upload
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must include target column)", type=["csv"])

# Sidebar - Model Selection
st.sidebar.header("2. Model Selection")
model_option = st.sidebar.selectbox(
    "Choose ML Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

@st.cache_resource
def load_model(name):
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    return joblib.load(filename)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'target' not in df.columns:
        st.error("Error: The uploaded CSV must contain a column named 'target'.")
    else:
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        # Load Model
        model = load_model(model_option)
        
        # Predictions
        preds = model.predict(X_test)
        
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = round(roc_auc_score(y_test, probs), 4)
        except Exception:
            auc = "N/A"
            
        # Metrics Calculation
        acc = round(accuracy_score(y_test, preds), 4)
        prec = round(precision_score(y_test, preds, zero_division=0), 4)
        rec = round(recall_score(y_test, preds, zero_division=0), 4)
        f1 = round(f1_score(y_test, preds, zero_division=0), 4)
        mcc = round(matthews_corrcoef(y_test, preds), 4)
        
        # Layout: Metrics Display
        st.subheader(f"📊 Model Performance: {model_option}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", acc)
        col2.metric("AUC Score", auc)
        col3.metric("Precision", prec)
        col4.metric("Recall", rec)
        col5.metric("F1 Score", f1)
        col6.metric("MCC Score", mcc)
        
        st.divider()
        
        # Layout: Confusion Matrix & Classification Report
        col_cm, col_rep = st.columns(2)
        
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        with col_rep:
            st.subheader("Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
else:
    st.info(" Please upload `test_data.csv` in the sidebar to view evaluation metrics.")