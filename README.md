# 🏦 Loan Approval Prediction System

## 🌐 Live Demo
👉 https://loan-approval-prediction-system-iltnbskd6bfywsopw4fokm.streamlit.app/
## githb repo link:
👉 https://github.com/wwwhimanshumaurya29-afk/loan-approval-prediction-system
---
## images
<img width="1920" height="1080" alt="Screenshot (188)" src="https://github.com/user-attachments/assets/6186c3bd-2492-468a-b020-92aba88d0698" />
<img width="1920" height="1080" alt="Screenshot (189)" src="https://github.com/user-attachments/assets/182122b2-25ee-490e-9e0b-7fbfe7b7347c" />



## 📌 Project Overview
A Machine Learning system that predicts whether a loan
application will be approved or rejected based on
applicant information such as income, credit history,
loan amount, and other factors.

---

## 👨‍💻 Project Details
| Detail | Info |
|--------|------|
| Project Type | Classification (ML) |
| Dataset | Kaggle Loan Prediction Dataset |
| Total Records | 614 loan applications |
| Target Variable | Loan_Status (Y=Approved, N=Rejected) |
| Best Model | Random Forest |
| Best Accuracy | ~80% |

---

## 📁 Folder Structure
LoanApproval_Project/

│

├── Dataset/

│   └── loan_data.csv              ← Raw dataset from Kaggle

│

├── Notebook/

│   └── loan.py                    ← Complete ML pipeline code

│

├── Model/

│   └── random_forest_model.pkl    ← Saved trained ML model

│

├── Streamlit_App/

│   └── app.py                     ← Web application code

│

├── Documentation/

│   ├── chart1_loan_status.png     ← Loan approval count

│   ├── chart2_gender.png          ← Gender analysis

│   ├── chart3_credit_history.png  ← Credit history impact

│   ├── chart4_education.png       ← Education impact

│   ├── chart5_income.png          ← Income distribution

│   ├── chart6_loanamount.png      ← Loan amount distribution

│   ├── chart7_property.png        ← Property area analysis

│   ├── chart8_roc_curve.png       ← ROC curve all models

│   ├── chart9_feature_importance  ← Feature importance

│   ├── chart10_model_comparison   ← Model comparison

│   ├── chart11_boxplots.png       ← Boxplots

│   ├── chart12_scatter.png        ← Scatter plot

│   └── chart13_heatmap.png        ← Correlation heatmap

│

└── README.md                      ← This file

---

## 🛠️ Technologies Used
| Technology | Purpose |
|-----------|---------|
| Python 3.14 | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models |
| XGBoost | Gradient boosting model |
| Imbalanced-learn | SMOTE for data balancing |
| Streamlit | Web application deployment |

---

## 📊 Dataset Information
- **Source:** Kaggle Loan Prediction Dataset
- **Link:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
- **Total Records:** 614
- **Total Features:** 13 columns

### Features Description:
| Column | Description | Type |
|--------|-------------|------|
| Loan_ID | Unique loan identifier | Dropped |
| Gender | Male / Female | Categorical |
| Married | Yes / No | Categorical |
| Dependents | Number of dependents | Categorical |
| Education | Graduate / Not Graduate | Categorical |
| Self_Employed | Yes / No | Categorical |
| ApplicantIncome | Monthly income of applicant | Numerical |
| CoapplicantIncome | Monthly income of co-applicant | Numerical |
| LoanAmount | Loan amount (thousands) | Numerical |
| Loan_Amount_Term | Loan repayment term (months) | Numerical |
| Credit_History | 1=Good, 0=Bad | Binary |
| Property_Area | Urban/Semiurban/Rural | Categorical |
| Loan_Status | Y=Approved, N=Rejected | TARGET |

---

## ⚙️ ML Pipeline

### Phase 1: Data Loading
- Loaded dataset with 614 records and 13 columns

### Phase 2: Data Cleaning
- Filled 149 missing values using Mode and Median
- Removed duplicate records
- Dropped irrelevant Loan_ID column

### Phase 3: Preprocessing
- Treated outliers using IQR method
- Normalized numerical columns using MinMaxScaler
- Encoded categorical columns using LabelEncoder

### Phase 4: EDA
- Created 13 visualization charts
- Univariate, Bivariate and Correlation Analysis

### Phase 5: Feature Engineering
- Created Total_Income (Applicant + Coapplicant)
- Created Income_Loan_Ratio (repayment capacity)
- Created EMI (monthly payment estimate)

### Phase 6: Model Building
- Balanced data using SMOTE (422 vs 422)
- Split data 80% train / 20% test
- Trained 4 ML models

### Phase 7: Model Evaluation
- Compared all models using multiple metrics

---

## 🤖 Model Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 69.82% | 0.77 | 0.70 | 0.69 |
| Decision Tree | 76.92% | 0.77 | 0.77 | 0.77 |
| Random Forest | 79.88% | 0.81 | 0.80 | 0.80 |
| XGBoost | 79.29% | 0.80 | 0.79 | 0.79 |

---

## 🏆 Best Model
Model    : Random Forest

Accuracy : 79.88%
---

## 🔑 Key Findings
Credit History is the most important feature (25.2%)
Income to Loan Ratio is 2nd most important (12.7%)
Total Income is 3rd most important (11.3%)
Semiurban areas have highest loan approval rates
Graduates get slightly more loan approvals
Good credit history = much higher chance of approval

---

## 🚀 How to Run

### Step 1 — Install Requirements:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost streamlit
```

### Step 2 — Run ML Pipeline:
```bash
python Notebook/loan.py
```

### Step 3 — Run Streamlit App:
```bash
streamlit run Streamlit_App/app.py
```

### Step 4 — Open Browser:
http://localhost:8501

---

## 📱 Streamlit App Features
- Input form for all applicant details
- Real time loan approval prediction
- Confidence percentage display
- Approval vs Rejection probability bar
- Clean and simple user interface
