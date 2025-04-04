import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


#  Load Dataset
df = pd.read_csv("encoded.csv")

#  Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


df['portfolio_value'] = df['portfolio_value'].clip(lower=0)
df['annual_income'] = df['annual_income'].clip(lower=0)


df['effective_savings'] = df['portfolio_value'] * df['savings_rate']
df['contribution_to_income_ratio'] = df['monthly_contribution'] / (df['annual_income'] / 12 + 1e-5)
df['volatility_risk_interaction'] = df['risk_appetite'] * df['market_volatility_index']
df['investment_density'] = df['portfolio_value'] / (df['investment_horizon_years'] + 1e-5)


X = df.drop(columns=['forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3'])
y = df[['forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3']]

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


base_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)


#  Make Predictions
y_pred = model.predict(X_test_scaled)

#  Evaluate Performance
for i, year in enumerate(['Year 1', 'Year 2', 'Year 3']):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\n Performance for {year}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Predict Future Portfolio Values
future_values = model.predict(X_test_scaled[:5])
print("\n Predicted Portfolio Values for Next 3 Years:\n", future_values)

# Feature Importance Plot
importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances,
    y=X.columns,    
    palette="viridis"
)
plt.title("Average Feature Importance (XGBoost across 3 years)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
