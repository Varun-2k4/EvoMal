import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import shap
from main_pipeline import run_pipeline

st.set_page_config(layout="wide")

# =====================================================
# Custom Styling
# =====================================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.navbar {
    background-color: #1f4e8c;
    padding: 15px 40px;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-title {
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.card {
    background-color: #f4f7fb;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Page State
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# Top Navbar
# =====================================================
st.markdown("""
<div class="navbar">
    <div class="nav-title">🛡 EvoMal-Net</div>
</div>
""", unsafe_allow_html=True)

# Right side navigation
left_space, nav1, nav2, nav3 = st.columns([6,1,1,1])

with nav1:
    if st.button("Dashboard"):
        st.session_state.page = "Dashboard"

with nav2:
    if st.button("Feature Selection"):
        st.session_state.page = "Feature"

with nav3:
    if st.button("Explainable AI"):
        st.session_state.page = "Explain"

# =====================================================
# HOME PAGE
# =====================================================
if st.session_state.page == "Home":

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h1 style='color:#1f4e8c;'>Welcome to EvoMal-Net</h1>
        <h4>Evolutionary AI-Driven Android Malware Detection System</h4>
        <p>Upload your dataset to begin detection using GA + Hybrid ML + SHAP Explainability.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded:
        if st.button("🚀 Get Started"):

            with st.spinner("Running GA + Hybrid Model..."):

                (
                    metrics,
                    model,
                    X_test,
                    selected_features,
                    selection_freq,
                    y_test
                ) = run_pipeline(uploaded)

                st.session_state.metrics = metrics
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.selected_features = selected_features

            st.session_state.page = "Dashboard"
            st.rerun()

# =====================================================
# DASHBOARD PAGE
# =====================================================
elif st.session_state.page == "Dashboard":

    if "metrics" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()

    metrics = st.session_state.metrics
    model = st.session_state.model
    y_test = st.session_state.y_test
    feature_names = st.session_state.selected_features

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Model Performance Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1 Score", f"{metrics['f1']:.4f}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    # -----------------------------
    # Feature Importance
    # -----------------------------
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top Feature Importance")

        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]

        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Malware vs Benign Pie
    # -----------------------------
    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Application Classification Distribution")

        safe = int((y_test == 0).sum())
        malware = int((y_test == 1).sum())
        total = safe + malware

        fig2, ax2 = plt.subplots()

        if total == 0:
            ax2.text(0.5, 0.5, "No Class Distribution Available",
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=12)
            ax2.axis("off")
        else:
            ax2.pie(
                [safe, malware],
                labels=["Benign", "Malware"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax2.axis("equal")

        st.pyplot(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FEATURE SELECTION PAGE
# =====================================================
elif st.session_state.page == "Feature":

    if "selected_features" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧬 Genetic Algorithm Feature Selection")

    st.write("### Selected Features After GA Optimization")

    for f in st.session_state.selected_features:
        st.write(f"• {f}")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================
elif st.session_state.page == "Explain":

    if "model" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Explainable AI (SHAP Summary)")

    try:
        model = st.session_state.model
        X_test = st.session_state.X_test
        feature_names = st.session_state.selected_features

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig = plt.figure()
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error("SHAP visualization could not be generated.")

    st.markdown("</div>", unsafe_allow_html=True)
