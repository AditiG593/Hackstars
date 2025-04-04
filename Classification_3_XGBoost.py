# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:30:44 2025

@author: SHRUTI-NIDHI
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


static_data = pd.read_csv("static_client_data.csv")
time_series_data = pd.read_csv("time_series_data.csv")
target_data = pd.read_csv("target_data.csv")


merged_df = time_series_data.merge(static_data, on="client_id", how="left")
merged_df = merged_df.merge(target_data, on="client_id", how="left")


df = merged_df.copy()


risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['risk_appetite'] = df['risk_appetite'].map(risk_mapping)
df['financial_knowledge_score'] = df['financial_knowledge_score'].astype(int)


categorical_cols = ['gender', 'employment_status', 'investment_goals', 'recommended_strategy']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Sort and drop unnecessary column
df = df.sort_values(by=['client_id', 'month'])
df.drop(columns=['preferred_asset_classes'], inplace=True)

# Create lag features
time_series_features = [
    'portfolio_value', 'equity_allocation_pct', 'fixed_income_allocation_pct',
    'monthly_contribution', 'market_volatility_index', 'macroeconomic_score', 'sentiment_index'
]

for feature in time_series_features:
    df[f'{feature}_lag_1'] = df.groupby('client_id')[feature].shift(1)
    df[f'{feature}_lag_3'] = df.groupby('client_id')[feature].shift(3)

df.ffill(inplace=True)

# Create trend features
df['portfolio_value_change_1m'] = df['portfolio_value'] - df['portfolio_value_lag_1']
df['equity_allocation_pct_change_1m'] = df['equity_allocation_pct'] - df['equity_allocation_pct_lag_1']
df['volatility_trend'] = df['market_volatility_index_lag_1'] - df['market_volatility_index_lag_3']

# Define target and features
classification_target = ['recommended_strategy_Aggressive', 'recommended_strategy_Balanced', 'recommended_strategy_Conservative']
features = df.drop(columns=['client_id', 'month'] + classification_target + [
    'forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3'
])

# Split data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    features, df[classification_target], test_size=0.2, random_state=42
)


y_train_cls_labels = y_train_cls.idxmax(axis=1)
y_test_cls_labels = y_test_cls.idxmax(axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_train_cls_encoded = label_encoder.fit_transform(y_train_cls_labels)
y_test_cls_encoded = label_encoder.transform(y_test_cls_labels)

# Train classifier
xgb_cls = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42)
xgb_cls.fit(X_train_cls, y_train_cls_encoded)

# Predict and evaluate
y_pred_cls = xgb_cls.predict(X_test_cls)
print("Classification Accuracy:", accuracy_score(y_test_cls_encoded, y_pred_cls))
print("Classification Report:\n", classification_report(y_test_cls_encoded, y_pred_cls))

# Map predictions back to original strategy names
y_pred_cls_labels = label_encoder.inverse_transform(y_pred_cls)
y_pred_cls_labels = [label.replace("recommended_strategy_", "") for label in y_pred_cls_labels]




