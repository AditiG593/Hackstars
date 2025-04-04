import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  Load Dataset
df = pd.read_csv("filled_data.csv")


#  Encode Categorical Features
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

X = df[['risk_appetite','portfolio_value','monthly_contribution','equity_allocation_pct','fixed_income_allocation_pct','market_volatility_index','annual_income','investment_horizon_years','savings_rate']]
y = df[['forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

for i, year in enumerate(['Year 1', 'Year 2', 'Year 3']):
    print(f"\n Performance for {year}:")
    print(f"MAE: {mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])):.2f}")


#  Predict Future Portfolio Values
future_values = model.predict(X_test_scaled[:5])  # Example for first 5 clients
print("\nPredicted Portfolio Values for Next 3 Years:\n", future_values)


