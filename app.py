import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import shap
from main_pipeline import run_pipeline

st.set_page_config(layout="wide")

# =====================================================
# API Friendly Name Mapping
# =====================================================

API_NAME_MAPPING = {

    # SMS & Communication
    "SEND_SMS": "SMS Sending Capability",
    "READ_SMS": "SMS Reading Permission",
    "RECEIVE_SMS": "SMS Receiving Permission",
    "WRITE_SMS": "SMS Modification Permission",
    "PROCESS_OUTGOING_CALLS": "Outgoing Call Interception",
    "BROADCAST_SMS": "SMS Broadcast Capability",

    # Device Identity
    "TelephonyManager.getDeviceId": "Device ID Access",
    "TelephonyManager.getSubscriberId": "Subscriber Identity Access",
    "TelephonyManager.getSimSerialNumber": "SIM Serial Number Access",
    "TelephonyManager.getNetworkOperator": "Network Operator Access",
    "GET_ACCOUNTS": "User Account Access",
    "MANAGE_ACCOUNTS": "Account Management Permission",
    "AUTHENTICATE_ACCOUNTS": "Account Authentication",

    # Storage & Filesystem
    "WRITE_EXTERNAL_STORAGE": "External Storage Write Access",
    "READ_EXTERNAL_STORAGE": "External Storage Read Access",
    "WRITE_CALL_LOG": "Call Log Modification",
    "WRITE_HISTORY_BOOKMARKS": "Browser History Modification",
    "chmod": "File Permission Modification",
    "chown": "File Ownership Change",
    "mount": "Filesystem Mount Operation",
    "remount": "Filesystem Remount Operation",

    # System Level
    "REBOOT": "Device Reboot Capability",
    "MOUNT_UNMOUNT_FILESYSTEMS": "Filesystem Control",
    "MODIFY_PHONE_STATE": "Phone State Modification",
    "STATUS_BAR": "Status Bar Manipulation",
    "INTERNAL_SYSTEM_WINDOW": "Internal System Window Access",

    # Location & Sensors
    "ACCESS_FINE_LOCATION": "Precise Location Access",
    "ACCESS_COARSE_LOCATION": "Approximate Location Access",
    "RECORD_AUDIO": "Microphone Access",

    # Dynamic Code / Obfuscation
    "Runtime.load": "Dynamic Native Code Loading",
    "Runtime.loadLibrary": "Dynamic Library Loading",
    "System.loadLibrary": "System Library Loading",
    "PathClassLoader": "Dynamic Class Loading",
    "defineClass": "Runtime Class Definition",
    "Ljavax.crypto.Cipher": "Cryptographic Cipher Usage",
    "Ljavax.crypto.spec.SecretKeySpec": "Secret Key Usage",

    # Persistence
    "android.intent.action.BOOT_COMPLETED": "Auto Start After Boot",
    "android.intent.action.PACKAGE_REMOVED": "App Removal Monitoring",
    "android.intent.action.SCREEN_ON": "Screen ON Event Detection",
    "android.intent.action.SCREEN_OFF": "Screen OFF Event Detection",

    # Binder Communication
    "bindService": "Service Binding Operation",
    "ServiceConnection": "Service Connection Handling",
    "transact": "Binder Transaction Execution",
    "attachInterface": "Interface Attachment",
    "getCallingUid": "Caller UID Identification",
    "getCallingPid": "Caller PID Identification",
}

def friendly_name(name):
    return API_NAME_MAPPING.get(name, name)

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
# Navbar
# =====================================================

st.markdown('<div class="navbar"><h2 style="color:white;">🛡 EvoMal-Net</h2></div>', unsafe_allow_html=True)

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
                st.session_state.selected_features = selected_features
                st.session_state.y_test = y_test

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
    selected_features = st.session_state.selected_features

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1 Score", f"{metrics['f1']:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    # Feature Importance
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top 10 Important Features")

        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]

        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([friendly_name(selected_features[i]) for i in indices])
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # Pie Chart
    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Benign vs Malware Distribution")

        safe = (y_test == 0).sum()
        malware = (y_test == 1).sum()

        fig2, ax2 = plt.subplots()
        ax2.pie([safe, malware], labels=["Benign", "Malware"], autopct="%1.1f%%")
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
    st.subheader("🧬 Selected Features After GA Optimization")

    for f in st.session_state.selected_features:
        st.write(f"• {friendly_name(f)}")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# EXPLAINABLE AI PAGE
# =====================================================

elif st.session_state.page == "Explain":

    if "model" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 SHAP Explainable AI")

    model = st.session_state.model
    X_test = st.session_state.X_test
    selected_features = st.session_state.selected_features

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    friendly_names = [friendly_name(f) for f in selected_features]

    fig = plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=friendly_names,
        show=False
    )
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)
