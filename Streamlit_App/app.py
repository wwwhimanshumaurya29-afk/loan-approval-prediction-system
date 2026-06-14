import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ✅ Dynamic path — works on ALL computers and Streamlit Cloud
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'Dataset', 'loan_data.csv')

# PAGE SETUP
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction System")
st.markdown("### Fill in your details to check loan eligibility")
st.divider()

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    # Clean
    df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df = df.drop('Loan_ID', axis=1)
    df = df.drop_duplicates()

    # Outlier treatment
    def treat_outliers(df, col):
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        return df

    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        df = treat_outliers(df, col)

    # Normalize
    scaler   = MinMaxScaler()
    num_cols = ['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode
    le       = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Dependents',
                'Education', 'Self_Employed',
                'Property_Area', 'Loan_Status']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Feature Engineering
    df['Total_Income']      = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 0.0001)
    df['EMI']               = df['LoanAmount'] / (df['Loan_Amount_Term'] + 0.0001)

    # Split & Balance
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42)

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# INPUT FORM
st.subheader("📝 Applicant Information")
col1, col2 = st.columns(2)

with col1:
    gender        = st.selectbox("Gender", ["Male", "Female"])
    married       = st.selectbox("Married", ["Yes", "No"])
    dependents    = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education     = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area",
                                  ["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income   = st.number_input("Applicant Monthly Income (₹)",
                                          min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)",
                                          min_value=0, value=0, step=500)
    loan_amount        = st.number_input("Loan Amount (in thousands ₹)",
                                          min_value=0, value=150, step=10)
    loan_term          = st.selectbox("Loan Term (months)",
                                       [360, 120, 180, 240, 300, 480, 60, 36, 12])
    credit_history     = st.selectbox("Credit History",
                                       ["Good (1.0)", "Bad (0.0)"])

st.divider()

# PREDICT BUTTON
if st.button("🔍 Check Loan Eligibility", use_container_width=True):

    # Process inputs
    gender_val        = 1 if gender == "Male" else 0
    married_val       = 1 if married == "Yes" else 0
    dependents_val    = {"0":0, "1":1, "2":2, "3+":3}[dependents]
    education_val     = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0
    property_val      = {"Urban":2, "Semiurban":1, "Rural":0}[property_area]
    credit_val        = 1.0 if "Good" in credit_history else 0.0

    # Normalize
    app_inc_norm   = min(applicant_income, 10171)   / 10171
    coapp_inc_norm = min(coapplicant_income, 5743)  / 5743
    loan_amt_norm  = min(loan_amount, 262)           / 262
    loan_term_norm = (loan_term - 12) / (480 - 12)

    # Feature Engineering
    total_income      = app_inc_norm + coapp_inc_norm
    income_loan_ratio = total_income / (loan_amt_norm + 0.0001)
    emi               = loan_amt_norm / (loan_term_norm + 0.0001)

    # Input dataframe
    input_data = pd.DataFrame([[
        gender_val, married_val, dependents_val, education_val,
        self_employed_val, app_inc_norm, coapp_inc_norm,
        loan_amt_norm, loan_term_norm, credit_val,
        property_val, total_income, income_loan_ratio, emi
    ]], columns=[
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Property_Area', 'Total_Income', 'Income_Loan_Ratio', 'EMI'
    ])

    # Predict
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence  = max(probability) * 100

    st.divider()

    # Show Result
    if prediction == 1:
        st.success("# ✅ LOAN APPROVED!")
        st.balloons()
        st.markdown(f"### Confidence: {confidence:.1f}%")
        st.markdown("**Your application meets the approval criteria.**")
    else:
        st.error("# ❌ LOAN REJECTED")
        st.markdown(f"### Confidence: {confidence:.1f}%")
        st.markdown("**Your application does not meet the approval criteria.**")

    # Probability Display
    st.divider()
    st.subheader("📊 Prediction Probability")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Approval Probability",
                  f"{probability[1]*100:.1f}%")
    with col2:
        st.metric("❌ Rejection Probability",
                  f"{probability[0]*100:.1f}%")
    st.progress(float(probability[1]))

# Footer
st.divider()
st.markdown("*Loan Approval Prediction System — ML Project*")