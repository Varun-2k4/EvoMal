import shap
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def show_shap(model, X):

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_sample = X.iloc[:200]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, show=False)

    st.pyplot(fig)
    plt.close(fig)
