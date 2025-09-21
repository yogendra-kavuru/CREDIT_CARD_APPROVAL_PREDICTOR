# INSTALLS
# pip install streamlit pandas scikit-learn seaborn matplotlib xgboost

# IMPORTS
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# -----------------------------
# LOAD SYNTHETIC DATASET
# -----------------------------
df = pd.read_csv('credit_card_dataset.csv')

# Encode categorical variables
df_encoded = df.copy()
df_encoded['EmploymentStatus'] = df_encoded['EmploymentStatus'].map({'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2})
df_encoded['ExistingLoan'] = df_encoded['ExistingLoan'].map({'No': 0, 'Yes': 1})

# Split features and label
X = df_encoded.drop('Approved', axis=1)
y = df_encoded['Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("💳 Credit Card Approval Prediction")
st.sidebar.header("Applicant Information")

# -----------------------------
# USER INPUT FUNCTION
# -----------------------------
def user_report():
    age = st.sidebar.slider("Age", 18, 65, 30)
    income = st.sidebar.slider("Annual Income ($)", 20000, 150000, 50000, step=1000)
    employment = st.sidebar.selectbox("Employment Status", ['Employed', 'Self-Employed', 'Unemployed'])
    years_at_job = st.sidebar.slider("Years at Current Job", 0, 30, 5)
    dependents = st.sidebar.slider("Number of Dependents", 0, 5, 1)
    dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    existing_loan = st.sidebar.selectbox("Existing Loan", ['Yes', 'No'])
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

    user_data = {
        'Age': age,
        'Income': income,
        'EmploymentStatus': employment,
        'YearsAtJob': years_at_job,
        'Dependents': dependents,
        'DebtToIncome': dti,
        'ExistingLoan': existing_loan,
        'CreditScore': credit_score
    }

    return pd.DataFrame(user_data, index=[0])

# -----------------------------
# USER DATA
# -----------------------------
user_data = user_report()
st.subheader("Applicant Data")
st.write(user_data)

# Encode user input
user_encoded = user_data.copy()
user_encoded['EmploymentStatus'] = user_encoded['EmploymentStatus'].map({'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2})
user_encoded['ExistingLoan'] = user_encoded['ExistingLoan'].map({'No': 0, 'Yes': 1})

# -----------------------------
# PREDICTION
# -----------------------------
xgb_result = xgb.predict(user_encoded)
st.subheader("Prediction Result:")
st.write("✅ Approved" if xgb_result[0] == 1 else "❌ Not Approved")

# -----------------------------
# VISUALISATION
# -----------------------------
st.subheader("Visualisation: Credit Score vs Income")
fig, ax = plt.subplots()
sns.scatterplot(x='CreditScore', y='Income', hue='Approved', data=df, palette='coolwarm', alpha=0.6, ax=ax)
sns.scatterplot(x=user_data['CreditScore'], y=user_data['Income'], color='black', s=200, ax=ax)
plt.title("Approval Distribution (Red=Not Approved, Blue=Approved)")
st.pyplot(fig)
