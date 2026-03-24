import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.figure_factory as ff

# ==========================
# CONFIGURATION DASHBOARD
# ==========================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Customer Churn Interactive Dashboard")
st.markdown("""
This dashboard allows you to:
- Predict customer churn
- Receive retention recommendations
- Explore interactive global statistics
- Understand predictions with SHAP explanations
""")

# ==========================
# CHARGER LE MODELE ET DATA
# ==========================
data = pd.read_csv("data.csv")  # dataset complet pour stats
model = joblib.load("churn_model.pkl")  # Pipeline avec ColumnTransformer + LogisticRegression

# Convertir Churn en numérique 0/1
if data["Churn"].dtype == object:
    data["Churn_binary"] = data["Churn"].map({"No":0,"Yes":1}).astype(int)
else:
    data["Churn_binary"] = data["Churn"].astype(int)

# ==========================
# SECTION 1: STATISTIQUES GLOBALES
# ==========================
st.header("📈 Global Churn Statistics")

col1, col2, col3 = st.columns(3)
total_customers = len(data)
churn_rate = data["Churn_binary"].mean() * 100
avg_monthly_charges = data["MonthlyCharges"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate (%)", f"{churn_rate:.2f}%")
col3.metric("Average Monthly Charges", f"${avg_monthly_charges:.2f}")

# Graphique interactif churn vs stay
st.subheader("Churn Distribution")
fig_churn = px.histogram(
    data, x="Churn", color="Churn",
    title="Number of Customers: Churn vs Stay",
    color_discrete_map={"Yes":"red","No":"green"}
)
st.plotly_chart(fig_churn, use_container_width=True)

# Graphique contrats vs churn
st.subheader("Contract Type vs Churn")
fig_contract = px.histogram(
    data, x="Contract", color="Churn",
    barmode="group",
    title="Contracts vs Churn",
    color_discrete_map={"Yes":"red","No":"green"}
)
st.plotly_chart(fig_contract, use_container_width=True)

# Heatmap interactive de corrélation
st.subheader("Feature Correlation Heatmap")
numeric_cols = data.select_dtypes(include=np.number)
corr_matrix = numeric_cols.corr()
fig_corr = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.columns),
    colorscale="Viridis",
    showscale=True,
    annotation_text=np.round(corr_matrix.values,2)
)
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================
# SECTION 2: PREDICTION CLIENT
# ==========================
st.header("🔮 Predict Individual Customer Churn")

st.subheader("Customer Input")
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Création DataFrame complet avec valeurs par défaut
input_data = pd.DataFrame({
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [tenure],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": [internet_service],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": [tech_support],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": [contract],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [monthly_charges * tenure]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    churn_prob = probability[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Customer likely to churn! Probability: {churn_prob:.2f}")
        if churn_prob > 0.8:
            st.warning("Recommendation: Offer 20% discount + Personal Call")
        elif churn_prob > 0.6:
            st.info("Recommendation: Offer loyalty program / free support")
        else:
            st.info("Recommendation: Send retention email")
    else:
        st.success(f"✅ Customer likely to stay. Probability: {churn_prob:.2f}")

    # SHAP explanation
    st.subheader("💡 Prediction Explanation (SHAP)")
    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        X_transformed = preprocessor.transform(input_data)
        explainer = shap.Explainer(classifier)
        shap_values = explainer(X_transformed)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning("SHAP explanation not available: " + str(e))

# ==========================
# SECTION 3: FEATURE IMPORTANCE GLOBALE
# ==========================
st.header("🔎 Global Feature Importance")
try:
    coefs = model.named_steps["classifier"].coef_[0]
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    fig_importance = px.bar(
        importance_df, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient", color_continuous_scale="Viridis",
        title="Feature Importance (Logistic Regression Coefficients)"
    )
    st.plotly_chart(fig_importance, use_container_width=True)
except Exception as e:
    st.warning("Feature importance not available: " + str(e))