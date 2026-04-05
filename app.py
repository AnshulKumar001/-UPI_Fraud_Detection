import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os
import warnings

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load model ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_model.pkl")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(model_path, "rb") as f:
                return pickle.load(f), None
    except Exception as e:
        return None, str(e)

model, load_err = load_model()

if load_err:
    st.error(f"⚠️ Model load failed: {load_err}")
    st.stop()   # ✅ important fix

# ── Feature names ──────────────────────────────────────
feature_names = [
    "step","type","amount",
    "oldbalanceOrg","newbalanceOrig",
    "oldbalanceDest","newbalanceDest","isFlaggedFraud"
]

# ── Dynamic Feature Importance (REAL) ──────────────────
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_ * 100
    FEATURE_IMPORTANCE = dict(zip(feature_names, importances))
else:
    FEATURE_IMPORTANCE = {name: 0 for name in feature_names}

# ── Constants ──────────────────────────────────────────
TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

TYPE_MAP = {
    "CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4
}

# ── Header ─────────────────────────────────────────────
st.title("🛡️ Fraud Detection System")

tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])

# ═══════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════
with tab1:

    step = st.number_input("Step", 1, 744, 1)
    tx_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
    amount = st.number_input("Amount", 0.0, value=10000.0)

    old_bal_orig = st.number_input("Old Balance Sender", 0.0, value=50000.0)
    new_bal_orig = st.number_input("New Balance Sender", 0.0, value=40000.0)

    old_bal_dest = st.number_input("Old Balance Receiver", 0.0, value=0.0)
    new_bal_dest = st.number_input("New Balance Receiver", 0.0, value=10000.0)

    is_flagged = st.selectbox("Flagged Fraud", [0,1])

    if st.button("Predict"):

        type_encoded = TYPE_MAP[tx_type]

        features = np.array([[step, type_encoded, amount,
                              old_bal_orig, new_bal_orig,
                              old_bal_dest, new_bal_dest,
                              is_flagged]])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        fraud_prob = proba[1] * 100
        legit_prob = proba[0] * 100

        if pred == 1:
            st.error(f"🚨 FRAUD ({fraud_prob:.2f}%)")
        else:
            st.success(f"✅ NOT FRAUD ({legit_prob:.2f}%)")

        # 🔥 Feature Importance UI
        st.subheader("Feature Importance")

        sorted_features = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: -x[1])[:5]

        for name, val in sorted_features:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
              <span style="width:150px;color:#aaa;">{name}</span>
              <div style="flex:1;height:6px;background:#222;border-radius:100px;">
                <div style="width:{val}%;height:100%;
                     background:linear-gradient(90deg,#0ea5e9,#38bdf8);
                     border-radius:100px;transition:0.8s;"></div>
              </div>
              <span style="width:40px;text-align:right;color:#666;">{val:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # 🔥 Top reason
        top_feature = max(FEATURE_IMPORTANCE, key=FEATURE_IMPORTANCE.get)
        st.info(f"🔥 Main factor: {top_feature}")

# ═══════════════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════════════
with tab2:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        required = feature_names
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df_proc = df.copy()
        df_proc['type'] = df_proc['type'].map(TYPE_MAP).fillna(3)

        preds = model.predict(df_proc.values)
        probas = model.predict_proba(df_proc.values)[:,1]

        df["Prediction"] = preds
        df["Fraud Probability"] = probas * 100

        st.dataframe(df.head())

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "results.csv"
        )