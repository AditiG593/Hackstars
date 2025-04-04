# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:38:16 2025

@author: SHRUTI-NIDHI
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/SHRUTI-NIDHI/OneDrive/Desktop/encoded.csv")

# Define Features & Target
features = [
    'risk_appetite', 'investment_goals_Education', 'investment_goals_Home Purchase', 
    'investment_goals_Retirement', 'investment_goals_Wealth Accumulation', 
    'financial_knowledge_score', 'portfolio_value'
]
target = 'recommended_strategy'

X = df[features].copy()  
y = df[target]

# Encode categorical features (if needed)
categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Ensure target is numeric
if y.dtype == 'object':  
    y = y.astype('category').cat.codes  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg_balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
log_reg_balanced.fit(X_train_scaled, y_train)

# Predict & Evaluate
y_pred_balanced = log_reg_balanced.predict(X_test_scaled)
print("\n✅ Logistic Regression (Balanced)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.2f}")