# app_professional.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load model and dataset
# -------------------------
model = joblib.load("churn_model.pkl")
df = pd.read_csv("data.csv")

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Customer Prediction", "Global Analysis", "Top Risk Customers"])

# -------------------------
# Header
# -------------------------
st.markdown("""
    <div style="background-color:blue; padding:20px; border-radius:10px; text-align:center;">
        <h1 style="color:white;"> Customer Churn Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# -------------------------
# Page 1 : Customer Prediction
# -------------------------
if page == "Customer Prediction":
    st.subheader("Enter Customer Information:")
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    # Build dataframe with all columns
    customer_data = pd.DataFrame({
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["No"],
        "Dependents": ["No"],
        "tenure": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": [internet_service],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
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
    st.markdown("""
    <style>
    .stButton>button:first-child {
        background-color: #0d6efd;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Predict Churn"):
        prediction = model.predict(customer_data)[0]
        probability = model.predict_proba(customer_data)[0][1]

        st.markdown(f"**Churn Probability:** {probability:.2f}")
        st.markdown(f"**Prediction:** {'Churn' if prediction==1 else 'No Churn'}")

        # Recommendations cards with colors
        if probability > 0.8:
            st.markdown("""
                <div style="padding:15px; border-radius:10px; background-color:#f8d7da; color:#842029; font-weight:bold;">
                    ⚠️ High Risk: Offer 20% discount + personal call
                </div>
            """, unsafe_allow_html=True)
        elif probability > 0.6:
            st.markdown("""
                <div style="padding:15px; border-radius:10px; background-color:#fff3cd; color:#664d03; font-weight:bold;">
                    ⚠️ Moderate Risk: Send retention email
                </div>
            """, unsafe_allow_html=True)
        elif probability > 0.4:
            st.markdown("""
                <div style="padding:15px; border-radius:10px; background-color:#d1e7dd; color:#0f5132; font-weight:bold;">
                    ✅ Medium Risk: Offer loyalty points
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="padding:15px; border-radius:10px; background-color:#cfe2ff; color:#084298; font-weight:bold;">
                    ✅ Low Risk: No action needed
                </div>
            """, unsafe_allow_html=True)

# -------------------------
# Page 2 : Global Analysis
# -------------------------
if page == "Global Analysis":
    total_clients = len(df)
    churn_count = df['Churn'].value_counts()

    # KPI cards
    col1, col2 = st.columns(2)
    col1.markdown(f"""
        <div style="background-color:#cce5ff; padding:20px; border-radius:10px; text-align:center;">
            <h3 style="color:#0d6efd;">Total Clients</h3>
            <h2 style="color:#0d6efd;">{total_clients}</h2>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
        <div style="background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center;">
            <h3 style="color:#842029;">Clients at Risk (Churn)</h3>
            <h2 style="color:#842029;">{churn_count.get('Yes',0)}</h2>
        </div>
    """, unsafe_allow_html=True)

    # Row 1: Pie + Histogram
    st.subheader("Churn Distribution & Monthly Charges")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(names=churn_count.index, values=churn_count.values, color=churn_count.index,
                      color_discrete_map={'No':'green','Yes':'red'},
                      title="Churn vs No Churn")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="MonthlyCharges", nbins=30, color="Churn", title="Monthly Charges Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Boxplot + Heatmap
    st.subheader("Tenure vs Churn & Correlation Heatmap")
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.box(df, x="Churn", y="tenure", color="Churn", title="Tenure by Churn")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        numeric_cols = df.select_dtypes(include='number')
        corr = numeric_cols.corr()
        fig4, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig4)

# -------------------------
# Page 3 : Top Risk Customers
# -------------------------
if page == "Top Risk Customers":
    st.subheader("Select Top N Risk Customers")
    top_n = st.number_input("Number of customers", min_value=1, max_value=500, value=10)

    df_model = pd.DataFrame({
        "gender": ["Male"]*len(df),
        "SeniorCitizen": [0]*len(df),
        "Partner": ["No"]*len(df),
        "Dependents": ["No"]*len(df),
        "tenure": df['tenure'],
        "PhoneService": ["Yes"]*len(df),
        "MultipleLines": ["No"]*len(df),
        "InternetService": df['InternetService'],
        "OnlineSecurity": ["No"]*len(df),
        "OnlineBackup": ["Yes"]*len(df),
        "DeviceProtection": ["No"]*len(df),
        "TechSupport": df['TechSupport'],
        "StreamingTV": ["No"]*len(df),
        "StreamingMovies": ["No"]*len(df),
        "Contract": df['Contract'],
        "PaperlessBilling": ["Yes"]*len(df),
        "PaymentMethod": ["Electronic check"]*len(df),
        "MonthlyCharges": df['MonthlyCharges'],
        "TotalCharges": df['MonthlyCharges']*df['tenure']
    })

    probabilities = model.predict_proba(df_model)[:,1]
    df['ChurnProbability'] = probabilities
    top_clients = df.sort_values(by='ChurnProbability', ascending=False).head(top_n)

    # Table with color for high risk
    def highlight_risk(row):
        color = ''
        if row['ChurnProbability'] > 0.8:
            color = 'background-color:#f8d7da; color:#842029; font-weight:bold;'
        elif row['ChurnProbability'] > 0.6:
            color = 'background-color:#fff3cd; color:#664d03; font-weight:bold;'
        return [color]*len(row)

    st.dataframe(top_clients[['customerID','tenure','MonthlyCharges','Contract','TechSupport','ChurnProbability']].style.apply(highlight_risk, axis=1))
    st.markdown("""
    <style>
    .stDownloadButton>button {
        background-color: #198754;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        width: 100%;
    }
    .stDownloadButton>button:hover {
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)
    # Export CSV button
    csv = top_clients.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Top Risk Customers",
        data=csv,
        file_name='top_risk_customers.csv',
        mime='text/csv',
    )