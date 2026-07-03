import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="AI Farming Assistant",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
try:
    model = joblib.load("crop_model.pkl")
except:
    st.error("❌ crop_model.pkl file not found.")
    st.stop()

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🌾 AI Farming Assistant")
st.write("Smart Crop Recommendation using Soil and Weather Conditions")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
menu = st.sidebar.selectbox(
    "Select Option",
    [
        "Home",
        "Crop Recommendation",
        "Cultivation Records",
        "AI Farming Assistant",
        "Dataset Analysis"
    ]
)

# ==================================================
# HOME
# ==================================================
if menu == "Home":

    st.header("Welcome Farmer 👨‍🌾")

    st.info("""
    This application can:

    ✔ Recommend crops

    ✔ Analyze soil conditions

    ✔ Store cultivation records

    ✔ Provide farming suggestions

    ✔ Analyze crop dataset
    """)

# ==================================================
# CROP RECOMMENDATION
# ==================================================
elif menu == "Crop Recommendation":

    st.header("🌱 Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", 0, 150, 90)
        P = st.number_input("Phosphorus (P)", 0, 150, 42)
        K = st.number_input("Potassium (K)", 0, 150, 43)
        temperature = st.number_input(
            "Temperature (°C)",
            0.0,
            50.0,
            25.0
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            0.0,
            100.0,
            80.0
        )

        ph = st.number_input(
            "pH Value",
            0.0,
            14.0,
            6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            0.0,
            500.0,
            200.0
        )

    if st.button("Recommend Crop"):

        data = np.array([[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]])

        prediction = model.predict(data)[0]

        st.success(
            f"🌱 Recommended Crop: {str(prediction).upper()}"
        )

        # Save cultivation record
        record = pd.DataFrame({
            "Date": [datetime.now()],
            "N": [N],
            "P": [P],
            "K": [K],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "pH": [ph],
            "Rainfall": [rainfall],
            "Recommended Crop": [prediction]
        })

        file = "cultivation_records.csv"

        if os.path.exists(file):
            record.to_csv(
                file,
                mode="a",
                header=False,
                index=False
            )
        else:
            record.to_csv(
                file,
                index=False
            )

        st.success("✅ Record saved successfully.")

# ==================================================
# CULTIVATION RECORDS
# ==================================================
elif menu == "Cultivation Records":

    st.header("📋 Cultivation Records")

    file = "cultivation_records.csv"

    if os.path.exists(file):

        df = pd.read_csv(file)

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "Download Records",
            df.to_csv(index=False),
            "records.csv",
            "text/csv"
        )

    else:
        st.warning("No records available.")

# ==================================================
# AI FARMING ASSISTANT
# ==================================================
elif menu == "AI Farming Assistant":

    st.header("🤖 Farming Assistant")

    crop = st.text_input(
        "Enter Crop Name"
    ).lower()

    if st.button("Get Advice"):

        tips = {

            "rice":
            """
            🌾 Rice:
            • Maintain standing water
            • Temperature: 20-35°C
            • Apply nitrogen fertilizer
            • Monitor pests
            """,

            "wheat":
            """
            🌾 Wheat:
            • Requires cool weather
            • Moderate irrigation
            • Use nitrogen fertilizer
            • Prevent fungal diseases
            """,

            "cotton":
            """
            🌱 Cotton:
            • Warm climate
            • Good drainage
            • Potassium fertilizer
            • Monitor bollworms
            """,

            "maize":
            """
            🌽 Maize:
            • Requires sunlight
            • Balanced fertilizer
            • Maintain soil moisture
            • Remove weeds
            """
        }

        if crop in tips:
            st.success(tips[crop])
        else:
            st.info("""
            General Farming Advice:

            ✔ Soil testing

            ✔ Organic fertilizers

            ✔ Crop rotation

            ✔ Monitor rainfall

            ✔ Use proper irrigation
            """)

# ==================================================
# DATASET ANALYSIS
# ==================================================
elif menu == "Dataset Analysis":

    st.header("📊 Dataset Analysis")

    try:
        df = pd.read_csv("Crop_recommendation.csv")

        st.write(df.head())

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Crop Distribution")
        st.bar_chart(
            df["label"].value_counts()
        )

        st.subheader("Statistics")
        st.write(df.describe())

    except:
        st.error(
            "Crop_recommendation.csv not found."
        )