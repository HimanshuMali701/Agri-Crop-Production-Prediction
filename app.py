import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# IMPORTANT: set_page_config must be the first Streamlit command
# ----------------------------
st.set_page_config(page_title="Crop Production Predictor", page_icon="🌾", layout="wide")

# ----------------------------
# Paths to model & encoders
# ----------------------------
MODEL_PATH = Path("data/agri_model.joblib")
LE_CROP_PATH = Path("data/le_crop.joblib")
LE_YEAR_PATH = Path("data/le_year.joblib")

# ----------------------------
# Load model & encoders (cached)
# ----------------------------
@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    le_crop = joblib.load(LE_CROP_PATH)
    le_year = joblib.load(LE_YEAR_PATH)
    return model, le_crop, le_year

try:
    model, le_crop, le_year = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# ----------------------------
# Metadata (from your dataset)
# ----------------------------
CROPS = [
 'Total Foodgrains','Rice','Wheat','Jowar','Bajra','Maize','Ragi','Small millets',
 'Barley','Coarse Cereals','Cereals','Gram','Arhar','Other Pulses','Total Pulses',
 'Total Non-Food grains ','Total Oilseeds','Groundnut','Sesamum','Rapeseed &Mustard',
 'Linseed','Castor seed','Safflower','Niger seed','Sunflower','Soyabean','Nine Oilseeds',
 'Coconut','Cotton seed','Total Fibers','Cotton(lint)','Jute','Mesta','Jute & Mesta',
 'Sannhamp ','Tea ','Coffee ','Rubber ','Total Spices','Black pepper','Dry chilies ',
 'Dry ginger ','Turmeric ','Arecanut  ','Cardamom ','Coriander','Garlic  ',
 'Total Fruits & Vegetables','Potato','Tapioca','Sweet potato  ','Onion','Banana  ',
 'Sugarcane','Tobacco'
]
YEARS = ['2006-07','2007-08','2008-09','2009-10','2010-11']

# ----------------------------
# Initialize session state for inputs so presets can set them
# ----------------------------
if 'crop' not in st.session_state:
    st.session_state['crop'] = CROPS[0]
if 'year' not in st.session_state:
    st.session_state['year'] = YEARS[0]
if 'area' not in st.session_state:
    st.session_state['area'] = 100.0
if 'yld' not in st.session_state:
    st.session_state['yld'] = 10.0

# ----------------------------
# SIDEBAR UI (presets, model info, feature importance)
# ----------------------------
with st.sidebar:
    st.header("Quick Actions & Model Info")

    st.subheader("Example presets")
    # When clicked, these buttons will set session_state values which the form picks up
    if st.button("Preset: Rice (2009-10)"):
        st.session_state['crop'] = "Rice"
        st.session_state['year'] = "2009-10"
        st.session_state['area'] = 120.0
        st.session_state['yld'] = 18.0
        st.success("Preset loaded: Rice 2009-10")

    if st.button("Preset: Wheat (2010-11)"):
        st.session_state['crop'] = "Wheat"
        st.session_state['year'] = "2010-11"
        st.session_state['area'] = 150.0
        st.session_state['yld'] = 20.0
        st.success("Preset loaded: Wheat 2010-11")

    if st.button("Preset: Maize (2008-09)"):
        st.session_state['crop'] = "Maize"
        st.session_state['year'] = "2008-09"
        st.session_state['area'] = 80.0
        st.session_state['yld'] = 14.0
        st.success("Preset loaded: Maize 2008-09")

    st.markdown("---")
    st.subheader("Model metrics (train)")
    # These values come from your training results — edit if you retrain
    st.write("- **MAE:** 5.72")
    st.write("- **RMSE:** 9.64")
    st.write("- **R²:** 0.973")

    st.markdown("---")
    st.subheader("Feature importance (top)")
    # Compute & display feature importances (model trained on ['Crop_encoded','Year_encoded','Area','Yield'])
    try:
        fi = model.feature_importances_
        features = ['Crop_encoded', 'Year_encoded', 'Area', 'Yield']
        fi_df = pd.DataFrame({'feature': features, 'importance': fi}).sort_values('importance', ascending=False)
        # Plot inside sidebar small
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(fi_df['feature'], fi_df['importance'])
        ax.set_title("Feature importances")
        ax.set_ylabel("Importance")
        ax.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.write("Could not compute feature importance:", e)

    
    st.write("Tip: use presets to demo quickly. Then click Predict in the main pane.")

# ----------------------------
# MAIN UI (wide)
# ----------------------------
st.title("🌾 Crop Production Predictor (India)")
st.write("Model trained on historical years 2006–2011. Predictions reflect patterns from that period.")

# Layout: left column form, right column info + feature importance plot larger
left_col, right_col = st.columns([2,1])

with left_col:
    # Use form so inputs can be changed then predict
    with st.form(key="predict_form"):
        st.subheader("Input details")
        crop = st.selectbox("Select Crop", CROPS, index=CROPS.index(st.session_state['crop']))
        year = st.selectbox("Select Historical Year (2006–2011)", YEARS, index=YEARS.index(st.session_state['year']))
        area = st.number_input("Area (as in dataset - Hectares)", min_value=0.0, value=float(st.session_state['area']))
        yld = st.number_input("Yield (Quintal / Hectare)", min_value=0.0, value=float(st.session_state['yld']))
        st.write("")  # spacing
        predict_button = st.form_submit_button("Predict Production")

with right_col:
    st.subheader("Model summary")
    st.write("RandomForestRegressor trained on historical production data.")
    st.write("Top features from training (visual):")
    # Larger plot for feature importance
    try:
        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.bar(fi_df['feature'], fi_df['importance'])
        ax2.set_title("Feature importances")
        ax2.set_ylabel("Importance")
        ax2.set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception:
        st.write("Feature importances not available.")

# ----------------------------
# PREDICTION LOGIC
# ----------------------------
if predict_button:
    # Update session_state so presets persist
    st.session_state['crop'] = crop
    st.session_state['year'] = year
    st.session_state['area'] = area
    st.session_state['yld'] = yld

    # Encode using saved LabelEncoders - ensure selection exists
    if crop not in list(le_crop.classes_):
        st.warning("Selected crop is not in the model encoder classes.")
    elif year not in list(le_year.classes_):
        st.warning("Selected year is not in the model encoder classes.")
    else:
        crop_encoded = int(le_crop.transform([crop])[0])
        year_encoded = int(le_year.transform([year])[0])
        X_input = np.array([[crop_encoded, year_encoded, float(area), float(yld)]])
        pred = float(model.predict(X_input)[0])

        # Units & uncertainty (MAE)
        dataset_unit = "Quintals"   # change if you know different
        mae_val = 5.725618181818187
        pred_tons = pred * 0.1
        mae_tons = mae_val * 0.1
        ci_low = pred - mae_val
        ci_high = pred + mae_val
        ci_low_t = ci_low * 0.1
        ci_high_t = ci_high * 0.1

        st.success(f"Predicted Production: **{pred:.2f} {dataset_unit}**  (≈ {pred_tons:.2f} Tons)")
        st.write(f"**Estimated uncertainty (±MAE):** {mae_val:.2f} {dataset_unit} (±{mae_tons:.2f} Tons)")
        st.write(f"Confidence interval (approx): **{ci_low:.2f} – {ci_high:.2f} {dataset_unit}**  (≈ {ci_low_t:.2f} – {ci_high_t:.2f} Tons)")
        st.info("Model trained on historical years 2006–2011. Predictions reflect patterns from that period.")
        st.markdown("**Inputs used:**")
        st.write(f"- Crop: {crop}")
        st.write(f"- Year: {year}")
        st.write(f"- Area: {area}")
        st.write(f"- Yield: {yld}")

# ----------------------------
# Footer
# ----------------------------

st.markdown("""
---
### 🌱 AgriPredict • Smart Agriculture Insights  
Enabling data-driven decisions for sustainable farming.  
Model trained using historical datasets (2006–2011).  
""")
