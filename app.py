import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ==========================================
# 1. APPLICATION CONFIGURATION & MODEL LOAD
# ==========================================
st.set_page_config(page_title="Zombie Survival Predictor", layout="centered", page_icon="🧟")

@st.cache_resource
def load_zombie_model():
    try:
        with open('zombie_survival_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception:
        return joblib.load('zombie_survival_model.pkl')

try:
    model_pipeline = load_zombie_model()
    st.success("🧟 Zombie Survival Model loaded successfully!")
except Exception as e:
    st.error("❌ Error loading model: Ensure 'zombie_survival_model.pkl' is uploaded in your GitHub repository.")
    st.stop()

# ==========================================
# 2. WEB APPLICATION USER INTERFACE (UI)
# ==========================================
st.title("🧟 Zombie Apocalypse Survival Predictor")
st.markdown("Enter your attributes below to see if you would survive the apocalypse or turn into a flesh-eating zombie!")

st.subheader("📋 Survivor Stats & Loadout")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Your Age:", min_value=1, max_value=120, value=25)
    physical_fitness = st.slider("Physical Fitness Score (1 = Weak, 10 = Athlete):", min_value=1, max_value=10, value=6)
    water_supply_days = st.number_input("Water Supply (Days):", min_value=0, max_value=100, value=7)

with col2:
    weapon_type = st.selectbox("Choose Your Primary Weapon:", options=['Shotgun', 'Melee', 'Frying Pan', 'None'])
    location = st.selectbox("Current Location Zone:", options=['Urban', 'Suburbs', 'Rural'])

# ==========================================
# 3. ROBUST PREDICTION LOGIC
# ==========================================
if st.button("🏹 Test Your Survival Chances", type="primary"):
    
    # User input structure matching the model
    user_inputs = {
        'Age': [age],
        'Physical_Fitness_Score': [physical_fitness],
        'Weapon_Type': [weapon_type],
        'Water_Supply_Days': [water_supply_days],
        'Location': [location]
    }
    
    try:
        # Check original features list from pipeline dynamically
        original_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
    except AttributeError:
        original_features = ['Age', 'Physical_Fitness_Score', 'Weapon_Type', 'Water_Supply_Days', 'Location']

    # Handle feature ordering automatically
    full_input_dict = {}
    for col in original_features:
        if col in user_inputs:
            full_input_dict[col] = user_inputs[col]
        else:
            full_input_dict[col] = [np.nan]

    input_df = pd.DataFrame(full_input_dict)[list(original_features)]
    
    try:
        # Run prediction
        prediction = model_pipeline.predict(input_df)
        prediction_prob = model_pipeline.predict_proba(input_df)[0][1] # Probability of surviving
        
        st.markdown("---")
        
        if prediction[0] == 1:
            st.balloons()
            st.success(f"🎉 **YOU SURVIVED!** You managed to beat the odds and stay alive.")
            st.metric(label="📊 Survival Probability", value=f"{prediction_prob * 100:.1f}%")
            st.info("💡 Tip: Keeping your water supply high and staying in Rural zones will ensure your ongoing survival!")
        else:
            st.snow()
            st.error(f"💀 **YOU TURNED!** Brains... You became a zombie.")
            st.metric(label="📊 Survival Probability", value=f"{prediction_prob * 100:.1f}%")
            st.warning("⚠️ Warning: Your weapon choice or fitness level was not enough to handle the urban zombie swarm!")
            
    except Exception as prediction_error:
        st.error(f"⚠️ Prediction Matrix Error: {str(prediction_error)}")
