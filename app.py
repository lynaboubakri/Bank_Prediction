import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- Page Setup ---
st.set_page_config(
    page_title="East Africa Banking Access Predictor",
    page_icon="🏦",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .prediction-header {
        font-size: 1.2rem !important;
        color: #2e86ab !important;
    }
    .confidence-badge {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🏦 East Africa Banking Access Predictor")
st.markdown("""
*Predicting financial inclusion across Rwanda, Tanzania, Kenya, and Uganda using machine learning*
""")


# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model_path = Path(__file__).parent / "best_rf_model.pkl"
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"""
        **Model Loading Error**: {e}

        Troubleshooting:
        1. Ensure 'best_rf_model.pkl' is in the same folder as this app
        2. Verify the file is not corrupted
        3. Check Python version compatibility (requires 3.8+)
        """)
        st.stop()


model = load_model()

# --- Country Mapping ---
COUNTRY_MAP = {
    1: "Rwanda",
    2: "Tanzania",
    3: "Kenya",
    4: "Uganda"
}

# --- Input Form ---
with st.form("prediction_form"):
    st.subheader("Demographic Information")

    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox(
            "Country",
            options=list(COUNTRY_MAP.keys()),
            format_func=lambda x: COUNTRY_MAP[x]
        )
        age = st.number_input("Age", min_value=12, max_value=100, value=30)
        gender = st.radio(
            "Gender",
            options=[1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female",
            horizontal=True
        )

    with col2:
        household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
        location_type = st.radio(
            "Residence Area",
            options=[1, 2],
            format_func=lambda x: "Urban" if x == 1 else "Rural",
            horizontal=True
        )
        cellphone_access = st.radio(
            "Has Cellphone Access",
            options=[1, 2],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True
        )

    st.subheader("Household & Employment Details")

    col3, col4 = st.columns(2)

    with col3:
        relationship = st.selectbox(
            "Relationship to Household Head",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "Head of Household",
                "Spouse",
                "Child",
                "Parent",
                "Other relative",
                "Other non-relatives"
            ][x - 1]
        )

        marital_status = st.selectbox(
            "Marital Status",
            options=[1, 2, 3, 4],
            format_func=lambda x: ["Married", "Single", "Divorced", "Widowed"][x - 1]
        )

    with col4:
        education_level = st.select_slider(
            "Education Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: ["None", "Primary", "Secondary", "Vocational", "Tertiary"][x - 1]
        )

        job_type = st.selectbox(
            "Primary Occupation",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            format_func=lambda x: [
                "Self employed",
                "Informally employed",
                "Farming and Fishing",
                "Remittance Dependent",
                "Other Income",
                "Formally employed Private",
                "No Income",
                "Formally employed Government",
                "Government Dependent",
                "Don't Know/Refuse to answer"
            ][x - 1]
        )

    submitted = st.form_submit_button(
        "Predict Banking Status",
        type="primary",
        use_container_width=True
    )

# --- Prediction Logic ---
if submitted:
    input_data = pd.DataFrame([[
        country, 0,  # Year placeholder
        location_type, cellphone_access,
        household_size, age, gender, relationship,
        marital_status, education_level, job_type
    ]], columns=[
        'country', 'year',
        'location_type', 'cellphone_access',
        'household_size', 'age_of_respondent',
        'gender_of_respondent', 'relationship_with_head',
        'marital_status', 'education_level', 'job_type'
    ])

    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        # Results Display
        st.divider()
        st.subheader("Prediction Result", anchor=False)

        result_col, conf_col = st.columns(2)
        with result_col:
            if prediction == 1:
                st.markdown(
                    f'<p class="prediction-header">✅ <strong>Prediction:</strong> Likely to have bank account</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p class="prediction-header">❌ <strong>Prediction:</strong> Unlikely to have bank account</p>',
                    unsafe_allow_html=True
                )

        with conf_col:
            confidence = max(proba) * 100
            st.markdown(
                f'<div class="confidence-badge">'
                f'<strong>Confidence Level:</strong> {confidence:.1f}%'
                f'</div>',
                unsafe_allow_html=True
            )

        # Probability Visualization
        with st.expander("View Detailed Probabilities"):
            tab1, tab2 = st.tabs(["📊 Chart", "📝 Numbers"])

            with tab1:
                proba_df = pd.DataFrame({
                    "Status": ["No Account", "Has Account"],
                    "Probability": [proba[0], proba[1]]
                })
                st.bar_chart(proba_df, x="Status", y="Probability")

            with tab2:
                st.write(f"- **Probability of having account:** {proba[1] * 100:.1f}%")
                st.write(f"- **Probability of no account:** {proba[0] * 100:.1f}%")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
        st.info("Please check your input values and try again")

# --- About Section ---
with st.sidebar:
    st.header("About This App", anchor=False)
    st.markdown("""
    **Banking Access Predictor** is a machine learning application that estimates the likelihood of individuals in East Africa having formal bank accounts based on demographic and socioeconomic factors.

    ### Key Features:
    - Predicts financial inclusion status
    - Covers Rwanda, Tanzania, Kenya, and Uganda
    - Uses Random Forest machine learning model
    - Provides confidence estimates

    ### Methodology:
    - Trained on nationally representative surveys
    - 90%+ accuracy on test data
    - Regularly updated with new data

    ### Intended Use:
    - For research and policy analysis
    - Financial inclusion initiatives
    - Academic studies

    *Developed with Python using Streamlit*
    """)

    st.divider()
    st.markdown("""
    **Disclaimer**: Predictions are statistical estimates, not financial advice. Always verify with official sources.
    """)

# --- Footer ---
st.divider()
st.caption("© 2023 East Africa Financial Inclusion Initiative | v1.0.0")