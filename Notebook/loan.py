import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import collections
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE

# =============================================
# SETUP PATHS
# =============================================
os.chdir(r'C:\Users\HP\OneDrive\Desktop\LoanApproval_Project')

DOC_PATH   = r'C:\Users\HP\OneDrive\Desktop\LoanApproval_Project\Documentation'
MODEL_PATH = r'C:\Users\HP\OneDrive\Desktop\LoanApproval_Project\Model'
os.makedirs(DOC_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# =============================================
# PHASE 1: LOAD AND UNDERSTAND DATA
# =============================================
df = pd.read_csv(r'Dataset\loan_data.csv')

print("=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== SHAPE ===")
print(df.shape)
print("\n=== COLUMN INFO ===")
print(df.info())
print("\n=== STATISTICS ===")
print(df.describe())
print("\n=== LOAN STATUS COUNT ===")
print(df['Loan_Status'].value_counts())

# =============================================
# PHASE 2: DATA CLEANING
# =============================================
print("\n=== MISSING VALUES BEFORE CLEANING ===")
print(df.isnull().sum())

df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())
df = df.drop('Loan_ID', axis=1)
df = df.drop_duplicates()

print("\n=== MISSING VALUES AFTER CLEANING ===")
print(df.isnull().sum())
print("\n=== SHAPE AFTER CLEANING ===")
print(df.shape)
print("\n✅ PHASE 2 COMPLETE!")

# =============================================
# PHASE 3: OUTLIERS + NORMALIZE
# =============================================
print("\n=== TREATING OUTLIERS ===")

def treat_outliers(df, col):
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return df

for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    df = treat_outliers(df, col)
    print(f"✅ Outliers treated: {col}")

# =============================================
# PHASE 4: EDA — CHARTS 1 TO 7
# (Before encoding so we still have text labels)
# =============================================
print("\n=== PHASE 4: EDA CHARTS ===")
sns.set_style("whitegrid")

# CHART 1 - Loan Status Count
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', hue='Loan_Status',
              data=df, palette='Set2', legend=False)
plt.title('Loan Approval Count (Y=Approved, N=Rejected)')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart1_loan_status.png'))
plt.show()
print("✅ Chart 1 Done!")

# CHART 2 - Gender vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Loan_Status',
              data=df, palette='Set1')
plt.title('Gender vs Loan Status')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart2_gender.png'))
plt.show()
print("✅ Chart 2 Done!")

# CHART 3 - Credit History vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status',
              data=df, palette='Set2')
plt.title('Credit History vs Loan Status')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart3_credit_history.png'))
plt.show()
print("✅ Chart 3 Done!")

# CHART 4 - Education vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(x='Education', hue='Loan_Status',
              data=df, palette='Set1')
plt.title('Education vs Loan Status')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart4_education.png'))
plt.show()
print("✅ Chart 4 Done!")

# CHART 5 - Income Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['ApplicantIncome'], bins=40,
             kde=True, color='blue')
plt.title('Applicant Income Distribution')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart5_income.png'))
plt.show()
print("✅ Chart 5 Done!")

# CHART 6 - Loan Amount Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['LoanAmount'], bins=40,
             kde=True, color='green')
plt.title('Loan Amount Distribution')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart6_loanamount.png'))
plt.show()
print("✅ Chart 6 Done!")

# CHART 7 - Property Area vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(x='Property_Area', hue='Loan_Status',
              data=df, palette='Set2')
plt.title('Property Area vs Loan Status')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart7_property.png'))
plt.show()
print("✅ Chart 7 Done!")

# CHART 11 - Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].boxplot(df['ApplicantIncome'])
axes[0].set_title('Applicant Income')
axes[1].boxplot(df['LoanAmount'])
axes[1].set_title('Loan Amount')
axes[2].boxplot(df['CoapplicantIncome'])
axes[2].set_title('Coapplicant Income')
plt.suptitle('Boxplots After Outlier Treatment')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart11_boxplots.png'))
plt.show()
print("✅ Chart 11 Done!")

# CHART 12 - Scatter Plot
plt.figure(figsize=(8,5))
colors_scatter = ['red' if s == 'N' else 'green'
                  for s in df['Loan_Status']]
plt.scatter(df['ApplicantIncome'],
            df['LoanAmount'],
            c=colors_scatter, alpha=0.6)
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.title('Scatter: Income vs Loan Amount')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart12_scatter.png'))
plt.show()
print("✅ Chart 12 Done!")

print("\n✅ PHASE 4 EDA CHARTS DONE!")

# =============================================
# NOW ENCODE + NORMALIZE
# (After EDA charts so labels are still readable)
# =============================================
print("\n=== ENCODING CATEGORICAL COLUMNS ===")
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents',
            'Education', 'Self_Employed',
            'Property_Area', 'Loan_Status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
print("✅ Encoding Done!")

print("\n=== NORMALIZING NUMERICAL COLUMNS ===")
scaler  = MinMaxScaler()
num_cols = ['ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term']
df[num_cols] = scaler.fit_transform(df[num_cols])
print("✅ Normalization Done!")

# CHART 13 - Heatmap (after encoding)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart13_heatmap.png'))
plt.show()
print("✅ Chart 13 Done!")

# =============================================
# PHASE 5: FEATURE ENGINEERING
# =============================================
print("\n=== PHASE 5: FEATURE ENGINEERING ===")

df['Total_Income']      = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 0.0001)
df['EMI']               = df['LoanAmount'] / (df['Loan_Amount_Term'] + 0.0001)

print("✅ Created: Total_Income")
print("✅ Created: Income_Loan_Ratio")
print("✅ Created: EMI")
print(f"New shape: {df.shape}")
print("\n✅ PHASE 5 COMPLETE!")

# =============================================
# PREPARE DATA FOR ML
# =============================================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
print("✅ DATA PREPARATION COMPLETE!")

# =============================================
# PHASE 6: BUILD ML MODELS
# =============================================
results = {}

# MODEL 1: Logistic Regression
print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)
lr_acc   = accuracy_score(y_test, lr_pred)
results['Logistic Regression'] = lr_acc
print(f"Accuracy : {lr_acc*100:.2f}%")
print(classification_report(y_test, lr_pred))

# MODEL 2: Decision Tree
print("\n" + "="*50)
print("MODEL 2: DECISION TREE")
print("="*50)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_acc   = accuracy_score(y_test, dt_pred)
results['Decision Tree'] = dt_acc
print(f"Accuracy : {dt_acc*100:.2f}%")
print(classification_report(y_test, dt_pred))

# MODEL 3: Random Forest
print("\n" + "="*50)
print("MODEL 3: RANDOM FOREST")
print("="*50)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)
results['Random Forest'] = rf_acc
print(f"Accuracy : {rf_acc*100:.2f}%")
print(classification_report(y_test, rf_pred))

# MODEL 4: XGBoost
print("\n" + "="*50)
print("MODEL 4: XGBOOST")
print("="*50)
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred  = xgb_model.predict(X_test)
xgb_acc   = accuracy_score(y_test, xgb_pred)
results['XGBoost'] = xgb_acc
print(f"Accuracy : {xgb_acc*100:.2f}%")
print(classification_report(y_test, xgb_pred))

# =============================================
# PHASE 7: EVALUATION CHARTS
# =============================================

# CHART 8 - ROC Curve
plt.figure(figsize=(10,6))
models_dict = {
    'Logistic Regression': (lr_model, lr_pred),
    'Decision Tree'      : (dt_model, dt_pred),
    'Random Forest'      : (rf_model, rf_pred),
    'XGBoost'            : (xgb_model, xgb_pred)
}
for name, (model, pred) in models_dict.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart8_roc_curve.png'))
plt.show()
print("✅ Chart 8 Done!")

# CHART 9 - Feature Importance
plt.figure(figsize=(10,6))
feature_importance = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature',
            hue='Feature', data=feature_importance,
            palette='viridis', legend=False)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart9_feature_importance.png'))
plt.show()
print("✅ Chart 9 Done!")

# CHART 10 - Model Comparison
plt.figure(figsize=(8,5))
model_names  = list(results.keys())
model_scores = [s*100 for s in results.values()]
colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff']
bars = plt.bar(model_names, model_scores,
               color=colors, edgecolor='black', width=0.5)
for bar, score in zip(bars, model_scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{score:.2f}%', ha='center',
             fontweight='bold', fontsize=11)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(DOC_PATH, 'chart10_model_comparison.png'))
plt.show()
print("✅ Chart 10 Done!")

# =============================================
# MODEL COMPARISON SUMMARY
# =============================================
print("\n" + "="*50)
print("📊 ALL MODELS COMPARISON")
print("="*50)
for name, acc in sorted(results.items(),
                         key=lambda x: x[1],
                         reverse=True):
    print(f"{name:25s} → {acc*100:.2f}%")

best_model_name = max(results, key=results.get)
print(f"\n🏆 BEST MODEL: {best_model_name} "
      f"→ {results[best_model_name]*100:.2f}%")

# =============================================
# SAVE MODEL
# =============================================
model_save_path = os.path.join(MODEL_PATH, 'random_forest_model.pkl')
with open(model_save_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\n✅ Model saved to: {model_save_path}")

# =============================================
# FINAL SUMMARY
# =============================================
print("\n" + "="*55)
print("🎯 FINAL PROJECT SUMMARY")
print("="*55)
print(f"📁 Dataset       : 614 loan applications")
print(f"🧹 Cleaning      : Fixed missing values")
print(f"⚖️  Balancing     : SMOTE applied")
print(f"🔀 Split         : 80% train / 20% test")
print(f"🤖 Models Tried  : 4 models")
print(f"🏆 Best Model    : {best_model_name} "
      f"→ {results[best_model_name]*100:.2f}%")
print(f"📊 Charts Saved  : 13 PNG files")
print(f"📁 Charts Path   : {DOC_PATH}")
print(f"💾 Model Path    : {model_save_path}")
print("="*55)
print("\n🎉 PROJECT COMPLETE!")