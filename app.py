import streamlit as st
import pandas as pd
from inference import PredictiveMaintenanceInference

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Predictive Maintenance – Milling Machine",
    layout="centered",
    page_icon="🛠️"
)

# -----------------------
# Load model once
# -----------------------
@st.cache_resource
def load_pipeline():
    return PredictiveMaintenanceInference()

pipeline = load_pipeline()

# -----------------------
# Title
# -----------------------
st.title("🛠️ Predictive Maintenance System")
st.markdown(
    """
    This tool predicts **potential machine failure** based on real-time
    sensor readings from a milling machine.

    - Output: **Failure Probability**
    - Model: **ANN (binary classification)**
    """
)

st.divider()

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("⚙️ Prediction Settings")

threshold = st.sidebar.slider(
    "Failure decision threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Lower = earlier maintenance, Higher = fewer false alarms"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Recommended thresholds**
    - 0.3 → Conservative (early warning)
    - 0.5 → Balanced
    - 0.7 → Strict (only critical failures)
    """
)

# -----------------------
# Input form
# -----------------------
st.subheader("🔧 Machine Sensor Inputs")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature (K)", 250.0, 400.0, 300.0)
        process_temp = st.number_input("Process Temperature (K)", 250.0, 400.0, 310.0)

    with col2:
        rotational_speed = st.number_input("Rotational Speed (rpm)", 0.0, 5000.0, 1500.0)
        torque = st.number_input("Torque (Nm)", 0.0, 200.0, 40.0)
        tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 100.0)

    submit = st.form_submit_button("🔍 Predict Failure")

# -----------------------
# Prediction
# -----------------------
if submit:
    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear
    }])

    try:
        failure_prob, prediction = pipeline.predict(input_df, threshold)

        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric(
            label="Failure Probability",
            value=f"{failure_prob:.2%}"
        )

        if prediction == 1:
            st.error("⚠️ **FAILURE LIKELY** — Schedule maintenance!")
        else:
            st.success("✅ **No Failure Detected**")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


# -----------------------
# Batch prediction via CSV
# -----------------------

st.divider()
st.subheader("📁 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader(
    "Upload CSV file (exact 6-column schema required)",
    type=["csv"]
)

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)

        # -----------------------------
        # STRICT SCHEMA DEFINITION
        # -----------------------------
        COLUMN_MAP = {
            "Type": "Type",
            "Air temperature": "Air temperature",
            "Air temperature [K]": "Air temperature",
            "Process temperature": "Process temperature",
            "Process temperature [K]": "Process temperature",
            "Rotational speed": "Rotational speed",
            "Rotational speed [rpm]": "Rotational speed",
            "Torque": "Torque",
            "Torque [Nm]": "Torque",
            "Tool wear": "Tool wear",
            "Tool wear [min]": "Tool wear",
        }

        REQUIRED_CANONICAL = {
            "Type",
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
        }

        uploaded_cols = list(df_raw.columns)

        # -----------------------------
        # 1. COLUMN COUNT CHECK
        # -----------------------------
        if len(uploaded_cols) != 6:
            st.error(
                f"❌ CSV must contain EXACTLY 6 columns. "
                f"Found {len(uploaded_cols)}."
            )
            st.stop()

        # -----------------------------
        # 2. COLUMN NAME VALIDATION
        # -----------------------------
        unknown_cols = set(uploaded_cols) - set(COLUMN_MAP.keys())
        if unknown_cols:
            st.error(
                f"❌ Invalid column names detected:\n{unknown_cols}"
            )
            st.stop()

        # -----------------------------
        # 3. CANONICAL RENAMING
        # -----------------------------
        df = df_raw.rename(columns=COLUMN_MAP)

        # -----------------------------
        # 4. REQUIRED FEATURE CHECK
        # -----------------------------
        if set(df.columns) != REQUIRED_CANONICAL:
            st.error(
                "❌ CSV must contain exactly these features:\n"
                f"{REQUIRED_CANONICAL}"
            )
            st.stop()

        # -----------------------------
        # 5. PREVIEW
        # -----------------------------
        st.write("✅ Valid CSV schema detected")
        st.dataframe(df.head())

        # -----------------------------
        # 6. INFERENCE
        # -----------------------------
        results = pipeline.predict_batch(df, threshold)
        output_df = pd.concat([df, results], axis=1)

        st.divider()
        st.subheader("📊 Batch Prediction Results")
        st.dataframe(output_df)

        st.metric(
            "Predicted Failures",
            int(output_df["failure_prediction"].sum())
        )

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            csv,
            "maintenance_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"❌ CSV processing failed: {str(e)}")

# -----------------------
# Footer
# -----------------------
st.divider()
st.caption(
    "Model trained on milling machine sensor data • "
    "Use as decision support, not as sole safety system."
)
