import streamlit as st
import requests


# =========================================================
# Configuration
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Multi-Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #f0ad4e;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Title
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🩺 Multi-Disease Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based symptom-to-disease prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# Backend Connection
# =========================================================

@st.cache_data
def get_symptoms():

    try:

        response = requests.get(
            f"{API_URL}/symptoms",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return data["symptoms"]

    except requests.exceptions.RequestException:

        return None


symptoms = get_symptoms()


# =========================================================
# Check Backend
# =========================================================

if symptoms is None:

    st.error(
        "❌ Unable to connect to the FastAPI backend."
    )

    st.info(
        "Please start FastAPI using:\n\n"
        "`uvicorn backend.main:app --reload`"
    )

    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("🩺 About")

    st.write(
        "This application uses a Deep Learning "
        "ANN model to predict diseases based on "
        "selected symptoms."
    )

    st.divider()

    st.write(
        f"**Available Symptoms:** {len(symptoms)}"
    )

    st.write(
        "**Disease Classes:** 659"
    )

    st.write(
        "**Model:** Artificial Neural Network"
    )

    st.divider()

    st.warning(
        "This system is for educational and "
        "research purposes only. It is not a "
        "medical diagnosis."
    )


# =========================================================
# Symptom Selection
# =========================================================

st.subheader("🔍 Select Your Symptoms")

st.write(
    "Search and select the symptoms you are experiencing."
)


selected_symptoms = st.multiselect(
    "Symptoms",
    options=symptoms,
    placeholder="Search symptoms...",
    label_visibility="collapsed"
)


# =========================================================
# Selected Symptoms
# =========================================================

if selected_symptoms:

    st.subheader(
        f"Selected Symptoms ({len(selected_symptoms)})"
    )

    cols = st.columns(3)

    for i, symptom in enumerate(
        selected_symptoms
    ):

        with cols[i % 3]:

            st.info(
                f"✓ {symptom}"
            )


# =========================================================
# Buttons
# =========================================================

col1, col2 = st.columns(2)


with col1:

    predict_button = st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    )


with col2:

    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )


# =========================================================
# Reset
# =========================================================

if reset_button:

    st.cache_data.clear()

    st.rerun()


# =========================================================
# Prediction
# =========================================================

if predict_button:

    if not selected_symptoms:

        st.warning(
            "⚠️ Please select at least one symptom."
        )

    else:

        with st.spinner(
            "Analyzing symptoms..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "symptoms": selected_symptoms
                    },
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()


                # =========================================
                # API Error
                # =========================================

                if not result.get("success"):

                    st.error(
                        result.get(
                            "message",
                            "Prediction failed."
                        )
                    )


                # =========================================
                # Successful Prediction
                # =========================================

                else:

                    predictions = result[
                        "predictions"
                    ]


                    st.divider()

                    st.subheader(
                        "🎯 Top 5 Predicted Diseases"
                    )


                    # =====================================
                    # Display Predictions
                    # =====================================

                    for i, prediction in enumerate(
                        predictions,
                        start=1
                    ):

                        disease = prediction[
                            "disease"
                        ]

                        probability = prediction[
                            "probability"
                        ]

                        percentage = probability * 100


                        st.markdown(
                            f"""
                            <div class="prediction-card">

                            <h4>
                            {i}. {disease}
                            </h4>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        st.progress(
                            probability
                        )

                        st.write(
                            f"Confidence: "
                            f"**{percentage:.2f}%**"
                        )


                    # =====================================
                    # Disclaimer
                    # =====================================

                    st.markdown(
                        """
                        <div class="warning-box">

                        ⚠️ <b>Important:</b>

                        These predictions are generated
                        by a machine learning model and
                        should not be considered a medical
                        diagnosis. Please consult a
                        qualified healthcare professional
                        for medical advice.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to FastAPI."
                )

                st.info(
                    "Make sure the backend is running:"
                )

                st.code(
                    "uvicorn backend.main:app --reload"
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The prediction request timed out."
                )


            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ API Error: {e}"
                )