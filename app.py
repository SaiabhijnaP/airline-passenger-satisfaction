"""
Airline Passenger Satisfaction Prediction - Streamlit Web Application
A modern, interactive decision-support application and viva demonstration platform.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="SkyPulse | Airline Passenger Satisfaction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .badge-satisfied {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-dissatisfied {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Artifacts
@st.cache_resource
def load_trained_pipeline():
    pipeline_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_pipeline.joblib')
    if not os.path.exists(pipeline_path):
        return None
    return joblib.load(pipeline_path)

@st.cache_data
def load_metadata():
    meta_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_eda_summary():
    eda_path = os.path.join(os.path.dirname(__file__), 'outputs', 'eda_summary.json')
    if os.path.exists(eda_path):
        with open(eda_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_comparison():
    comp_path = os.path.join(os.path.dirname(__file__), 'outputs', 'model_comparison.csv')
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("✈️ SkyPulse AI")
st.sidebar.caption("Kaggle Playground Series (S6E10)")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home & Overview",
        "📊 EDA Dashboard",
        "🔮 Predict Satisfaction",
        "📁 Batch Prediction",
        "📈 Model Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Status:**
- Model: `Hist Gradient Boosting`
- Primary Metric: **ROC-AUC (0.9568)**
- Validation Accuracy: **92.38%**
""")


# -----------------------------------------------------------------------------
# PAGE 1: HOME & OVERVIEW
# -----------------------------------------------------------------------------
if page == "🏠 Home & Overview":
    st.markdown('<div class="main-header">✈️ Airline Passenger Satisfaction Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning decision-support platform grounded in Kaggle Playground Series Season 6 Episode 10</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    eda = load_eda_summary()
    metrics = load_model_metrics()

    with col1:
        st.metric(
            label="Total Dataset Size",
            value=f"{eda['total_rows']:,}" if eda else "699,635",
            help="Total passenger feedback records"
        )
    with col2:
        st.metric(
            label="Selected Model",
            value=metrics['best_model'] if metrics else "Hist Gradient Boosting",
            help="Best performing algorithm selected via cross-model validation"
        )
    with col3:
        roc_score = metrics['best_metrics']['ROC_AUC'] if metrics else 0.9568
        st.metric(
            label="Validation ROC-AUC",
            value=f"{roc_score:.4f}",
            help="Official competition primary evaluation metric"
        )
    with col4:
        acc_score = metrics['best_metrics']['Accuracy'] if metrics else 0.9238
        st.metric(
            label="Validation Accuracy",
            value=f"{acc_score*100:.2f}%",
            help="Stratified holdout validation accuracy"
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 Project Problem Statement")
        st.write("""
        Customer satisfaction in the commercial aviation industry is a core differentiator driving repeat bookings, customer lifetime value, and brand loyalty. 
        Traditional post-flight surveys often yield delayed insights, preventing airlines from proactively addressing pain points.
        
        This project builds an end-to-end, reproducible Machine Learning pipeline to predict whether an airline passenger is **Satisfied** or **Dissatisfied** based on demographic attributes, travel characteristics, delays, and 14 detailed inflight and ground touchpoint ratings.
        """)

        st.subheader("🎯 Key Business & Technical Objectives")
        st.markdown("""
        - **Accurate Inflight Experience Scoring:** Predict customer sentiment with over 92% classification accuracy and 0.956+ ROC-AUC.
        - **Touchpoint Prioritization:** Identify which operational services (e.g., Inflight Wi-Fi, Online Boarding, Seat Comfort) show the strongest statistical association with passenger contentment.
        - **Production-Ready Inference:** Provide instant single-passenger interactive predictions and high-throughput batch CSV scoring.
        - **Zero Data Leakage:** Ensure rigorous preprocessing encapsulation utilizing Scikit-Learn `ColumnTransformer` and `Pipeline`.
        """)

    with col_right:
        st.subheader("🔍 Dataset Overview")
        st.markdown("""
        - **Source:** Kaggle Playground Series S6E10
        - **Training Records:** 699,635 rows
        - **Official Test Set:** 299,844 rows
        - **Target Variable:** `satisfaction` (Binary: `True`/`False`)
        - **Feature Types:**
            - **Demographic:** Gender, Age, Customer Type (Loyal vs Disloyal)
            - **Flight Info:** Type of Travel, Class, Flight Distance
            - **Operational:** Departure Delay, Arrival Delay
            - **Survey Ratings:** 14 distinct touchpoints rated on a 0-5 scale
        """)
        st.info("💡 **Viva Tip:** All preprocessing, models, and predictions are pre-compiled and cached. Submissions strictly match official Kaggle requirements.")


# -----------------------------------------------------------------------------
# PAGE 2: EDA DASHBOARD
# -----------------------------------------------------------------------------
elif page == "📊 EDA Dashboard":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistical distributions, customer demographics, and touchpoint correlation patterns</div>', unsafe_allow_html=True)

    eda = load_eda_summary()

    if eda:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dissatisfied Passengers", f"{eda['dissatisfied_count']:,}", f"{(1 - eda['satisfied_rate'])*100:.1f}%")
        c2.metric("Satisfied Passengers", f"{eda['satisfied_count']:,}", f"{eda['satisfied_rate']*100:.1f}%")
        c3.metric("Missing Arrival Delays", f"{eda['missing_values'].get('Arrival Delay in Minutes', 292):,}", "0.04% of total")
        c4.metric("Duplicate Records", f"{eda['duplicate_rows']:,}", "Clean dataset")

    tabs = st.tabs([
        "🎯 Target & Demographics",
        "⭐ Service Touchpoints",
        "⏱️ Delays & Distances",
        "🔥 Correlation Heatmap",
        "💡 Key Insights"
    ])

    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    with tabs[0]:
        st.subheader("1. Target Distribution & Demographic Breakdowns")
        col_a, col_b = st.columns(2)
        p1 = os.path.join(outputs_dir, '01_target_distribution.png')
        p2 = os.path.join(outputs_dir, '02_demographics_and_satisfaction.png')
        if os.path.exists(p1):
            col_a.image(p1, caption="Figure 1: Balanced Target Distribution (44.36% Satisfied vs 55.64% Dissatisfied)", use_container_width=True)
        if os.path.exists(p2):
            col_b.image(p2, caption="Figure 2: Satisfaction Across Travel Class, Purpose, and Customer Loyalty", use_container_width=True)

    with tabs[1]:
        st.subheader("2. Passenger Service Ratings Comparison")
        p3 = os.path.join(outputs_dir, '03_service_ratings_analysis.png')
        if os.path.exists(p3):
            st.image(p3, caption="Figure 3: Mean Service Ratings for Satisfied vs Dissatisfied Passengers (Ordered by Gap)", use_container_width=True)

    with tabs[2]:
        st.subheader("3. Flight Delays & Distance Distributions")
        col_c, col_d = st.columns(2)
        p4 = os.path.join(outputs_dir, '04_numerical_distributions.png')
        p5 = os.path.join(outputs_dir, '05_delays_analysis.png')
        if os.path.exists(p4):
            col_c.image(p4, caption="Figure 4: Age & Flight Distance Densities", use_container_width=True)
        if os.path.exists(p5):
            col_d.image(p5, caption="Figure 5: Departure and Arrival Delay Correlational Patterns", use_container_width=True)

    with tabs[3]:
        st.subheader("4. Feature Correlation Matrix")
        p6 = os.path.join(outputs_dir, '06_correlation_heatmap.png')
        if os.path.exists(p6):
            st.image(p6, caption="Figure 6: Complete Correlation Heatmap of Numerical and Rating Features", use_container_width=True)

    with tabs[4]:
        st.subheader("💡 Strategic Findings from Exploratory Data Analysis")
        st.markdown("""
        1. **Class Disparity:**
           - Over **70% of Business Class** travelers report satisfaction, whereas less than **25% of Economy Class** passengers are satisfied.
           - Upgrading seat comfort and legroom directly mitigates customer friction.
        2. **The "Digital Gatekeeper" Touchpoints:**
           - **Online Boarding** has the single strongest correlation with passenger satisfaction ($r = +0.560$) and the largest rating gap (+1.45 points on a 5-point scale).
           - **Inflight Entertainment** ($r = +0.431$) and **Inflight Wi-Fi** are decisive drivers of long-haul travel contentment.
        3. **Delay Impact vs Service Quality:**
           - While flight delays (Departure & Arrival) negatively affect customer sentiment, operational flight delays alone explain only a small fraction of overall variance.
           - Excellent cabin service and smooth digital check-in can salvage passenger loyalty even on delayed flights.
        """)


# -----------------------------------------------------------------------------
# PAGE 3: PREDICT SATISFACTION (SINGLE INFERENCE)
# -----------------------------------------------------------------------------
elif page == "🔮 Predict Satisfaction":
    st.markdown('<div class="main-header">🔮 Predict Passenger Satisfaction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive simulation tool to predict customer sentiment for an individual flight record</div>', unsafe_allow_html=True)

    pipeline = load_trained_pipeline()
    if pipeline is None:
        st.error("⚠️ Trained model pipeline not found. Please execute `python train.py` first.")
        st.stop()

    # Preset Quick Loaders for viva demos
    st.markdown("#### ⚡ Demo Quick Presets")
    col_pre1, col_pre2, col_pre3 = st.columns([1, 1, 2])
    preset_choice = None
    if col_pre1.button("✨ Load Business Traveler (Likely Satisfied)"):
        preset_choice = "satisfied"
    if col_pre2.button("⚠️ Load Economy Traveler (Likely Dissatisfied)"):
        preset_choice = "dissatisfied"

    # Default values based on preset
    if preset_choice == "satisfied":
        def_gender, def_cust, def_age, def_type, def_class, def_dist = "Female", "Loyal Customer", 42, "Business travel", "Business", 1850
        def_wifi, def_time, def_book, def_gate, def_food, def_board = 5, 4, 4, 4, 4, 5
        def_seat, def_ent, def_onboard, def_leg, def_bag, def_check, def_clean = 5, 5, 5, 5, 5, 5, 5
        def_dep_del, def_arr_del = 0, 0.0
    elif preset_choice == "dissatisfied":
        def_gender, def_cust, def_age, def_type, def_class, def_dist = "Male", "disloyal Customer", 24, "Personal Travel", "Eco", 450
        def_wifi, def_time, def_book, def_gate, def_food, def_board = 1, 2, 1, 2, 1, 1
        def_seat, def_ent, def_onboard, def_leg, def_bag, def_check, def_clean = 2, 1, 2, 1, 2, 2, 1
        def_dep_del, def_arr_del = 50, 60.0
    else:
        def_gender, def_cust, def_age, def_type, def_class, def_dist = "Female", "Loyal Customer", 35, "Business travel", "Business", 1200
        def_wifi, def_time, def_book, def_gate, def_food, def_board = 4, 3, 3, 3, 3, 4
        def_seat, def_ent, def_onboard, def_leg, def_bag, def_check, def_clean = 4, 4, 4, 4, 4, 4, 4
        def_dep_del, def_arr_del = 0, 0.0

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 👤 Passenger & Flight Profile")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gender == "Female" else 1)
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"], index=0 if def_cust == "Loyal Customer" else 1)
            age = st.slider("Passenger Age (Years)", min_value=7, max_value=85, value=def_age)
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"], index=0 if def_type == "Business travel" else 1)
            flight_class = st.selectbox("Travel Class", ["Business", "Eco", "Eco Plus"], index=["Business", "Eco", "Eco Plus"].index(def_class))
            flight_distance = st.number_input("Flight Distance (Miles)", min_value=30, max_value=5000, value=def_dist, step=50)
            dep_delay = st.number_input("Departure Delay (Minutes)", min_value=0, max_value=1500, value=def_dep_del, step=5)
            arr_delay = st.number_input("Arrival Delay (Minutes)", min_value=0.0, max_value=1500.0, value=def_arr_del, step=5.0)

        with col2:
            st.markdown("##### ⭐ Service Quality Ratings (0: N/A, 1: Poor to 5: Excellent)")
            wifi = st.slider("Inflight Wi-Fi Service", 0, 5, def_wifi)
            online_boarding = st.slider("Online Boarding Experience", 0, 5, def_board)
            ease_booking = st.slider("Ease of Online Booking", 0, 5, def_book)
            seat_comfort = st.slider("Seat Comfort", 0, 5, def_seat)
            entertainment = st.slider("Inflight Entertainment", 0, 5, def_ent)
            onboard_service = st.slider("On-board Service", 0, 5, def_onboard)
            leg_room = st.slider("Leg Room Service", 0, 5, def_leg)
            cleanliness = st.slider("Cleanliness", 0, 5, def_clean)
            food_drink = st.slider("Food and Drink", 0, 5, def_food)
            baggage = st.slider("Baggage Handling", 1, 5, def_bag)
            checkin = st.slider("Check-in Service", 0, 5, def_check)
            gate_loc = st.slider("Gate Location Convenience", 0, 5, def_gate)
            time_conv = st.slider("Departure/Arrival Time Convenient", 0, 5, def_time)

        submitted = st.form_submit_button("🚀 Predict Passenger Satisfaction", use_container_width=True)

    if submitted:
        input_record = {
            'Gender': gender,
            'Customer Type': customer_type,
            'Age': age,
            'Type of Travel': travel_type,
            'Class': flight_class,
            'Flight Distance': flight_distance,
            'Inflight wifi service': wifi,
            'Departure/Arrival time convenient': time_conv,
            'Ease of Online booking': ease_booking,
            'Gate location': gate_loc,
            'Food and drink': food_drink,
            'Online boarding': online_boarding,
            'Seat comfort': seat_comfort,
            'Inflight entertainment': entertainment,
            'On-board service': onboard_service,
            'Leg room service': leg_room,
            'Baggage handling': baggage,
            'Checkin service': checkin,
            'Cleanliness': cleanliness,
            'Departure Delay in Minutes': dep_delay,
            'Arrival Delay in Minutes': arr_delay
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_record])
        prob_satisfied = float(pipeline.predict_proba(input_df)[0][1])
        prediction = pipeline.predict(input_df)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Outcome")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if prediction == 1:
                st.markdown('<div class="badge-satisfied">✅ SATISFIED PASSENGER</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="badge-dissatisfied">❌ DISSATISFIED PASSENGER</div>', unsafe_allow_html=True)

            st.write("")
            st.metric("Satisfaction Probability", f"{prob_satisfied*100:.2f}%")
            st.metric("Dissatisfaction Probability", f"{(1-prob_satisfied)*100:.2f}%")

        with col_res2:
            st.write("**Model Confidence Level:**")
            st.progress(prob_satisfied if prediction == 1 else (1 - prob_satisfied))

            if prediction == 1:
                st.success(f"The model predicts with **{prob_satisfied*100:.1f}% confidence** that this passenger will report a positive satisfaction experience.")
            else:
                st.warning(f"The model predicts with **{(1-prob_satisfied)*100:.1f}% confidence** that this passenger will report being dissatisfied. Consider proactive customer service intervention.")


# -----------------------------------------------------------------------------
# PAGE 4: BATCH PREDICTION
# -----------------------------------------------------------------------------
elif page == "📁 Batch Prediction":
    st.markdown('<div class="main-header">📁 High-Throughput Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload passenger manifest or feedback CSV files to generate predictions at scale</div>', unsafe_allow_html=True)

    pipeline = load_trained_pipeline()
    metadata = load_metadata()

    if pipeline is None:
        st.error("⚠️ Trained model pipeline not found. Please run `python train.py` first.")
        st.stop()

    st.markdown("""
    Upload a CSV file containing passenger records. The system validates all required columns, 
    imputes missing delay records automatically, and outputs predictions alongside individual probability scores.
    """)

    # Provide sample download button
    test_path = os.path.join(os.path.dirname(__file__), 'test.csv')
    if os.path.exists(test_path):
        sample_batch = pd.read_csv(test_path, nrows=50)
        csv_sample = sample_batch.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Batch CSV (50 Test Records)",
            data=csv_sample,
            file_name="sample_passenger_batch.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload Passenger Records CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded file contains **{len(batch_df):,} rows** and **{len(batch_df.columns)} columns**.")

            # Validate Columns
            req_cols = set(metadata['all_features'] if metadata else [
                'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                'Baggage handling', 'Checkin service', 'Cleanliness', 'Departure Delay in Minutes',
                'Arrival Delay in Minutes', 'Gender', 'Customer Type', 'Type of Travel', 'Class'
            ])
            missing = list(req_cols - set(batch_df.columns))

            if missing:
                st.error(f"❌ Input file is missing {len(missing)} required columns:\n`{missing}`")
            else:
                st.success("✅ Column validation successful! Processing predictions...")

                # Filter and order features
                X_batch = batch_df[list(req_cols)]
                preds = pipeline.predict(X_batch)
                probs = pipeline.predict_proba(X_batch)[:, 1]

                results_df = batch_df.copy()
                results_df['Predicted_Satisfaction'] = np.where(preds == 1, 'Satisfied', 'Dissatisfied')
                results_df['Satisfaction_Probability'] = np.round(probs, 4)
                results_df['Confidence'] = np.round(np.where(preds == 1, probs, 1 - probs) * 100, 2)

                # Summary Metrics
                sat_count = int((preds == 1).sum())
                dissat_count = int((preds == 0).sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Satisfied", f"{sat_count:,}", f"{sat_count/len(batch_df)*100:.1f}%")
                m2.metric("Predicted Dissatisfied", f"{dissat_count:,}", f"{dissat_count/len(batch_df)*100:.1f}%")
                m3.metric("Average Confidence", f"{results_df['Confidence'].mean():.2f}%")

                st.dataframe(results_df.head(100), use_container_width=True)

                # Download Results CSV
                output_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Complete Batch Predictions CSV",
                    data=output_csv,
                    file_name="passenger_satisfaction_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")


# -----------------------------------------------------------------------------
# PAGE 5: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-header">📈 Model Benchmarking & Performance Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparative analysis of Logistic Regression, Random Forest, and Hist Gradient Boosting</div>', unsafe_allow_html=True)

    comp_df = load_model_comparison()
    metrics = load_model_metrics()
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    if comp_df is not None:
        st.subheader("1. Validation Benchmark Leaderboard")
        st.dataframe(comp_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'], color='#D1FAE5'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Metrics Comparison Across Models")
        p7 = os.path.join(outputs_dir, '07_model_comparison.png')
        if os.path.exists(p7):
            st.image(p7, caption="Figure 7: Model Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)", use_container_width=True)

    with col2:
        st.subheader("3. Feature Importance Analysis")
        p9 = os.path.join(outputs_dir, '09_feature_importance.png')
        if os.path.exists(p9):
            st.image(p9, caption="Figure 8: Permutation Feature Importance (Validation Drop in ROC-AUC)", use_container_width=True)

    st.subheader("4. Confusion Matrices Across Models")
    p8 = os.path.join(outputs_dir, '08_confusion_matrices.png')
    if os.path.exists(p8):
        st.image(p8, caption="Figure 9: Confusion Matrices on 139,927 Stratified Holdout Samples", use_container_width=True)

    st.subheader("🎓 Viva & Technical Discussion Notes")
    st.markdown("""
    - **Why Hist Gradient Boosting is the Winning Model:**
      - **Highest Metric Score:** Outperformed Logistic Regression and Random Forest on the primary competition metric (**0.9568 ROC-AUC**).
      - **Training Efficiency:** Trains in **~5 seconds** across over half a million rows thanks to integer histogram binning (comparable to LightGBM).
      - **Compact Footprint:** Produces a lightweight joblib artifact (< 5 MB) suitable for real-time web inference.
    - **Causality vs Association Disclaimer:**
      - The permutation importance chart indicates that *Type of Travel*, *Inflight Wi-Fi Service*, and *Customer Loyalty* are the strongest predictors of satisfaction.
      - **Crucial Note:** These statistical associations identify critical leverage points for airline management, but do not imply direct mathematical causation without randomized control trials.
    """)
