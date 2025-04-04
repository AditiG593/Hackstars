
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# Load Dataset
df = pd.read_csv("encoded.csv")

# Encode Categorical Features
client_ids = df['client_id'].copy()
categorical_cols = df.select_dtypes(include=['object']).columns.drop('client_id')
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
df['client_id'] = client_ids

# Handle negative values
df['portfolio_value'] = df['portfolio_value'].clip(lower=0)
df['annual_income'] = df['annual_income'].clip(lower=0)

# Create Derived Features
df['effective_savings'] = df['portfolio_value'] * df['savings_rate']
df['contribution_to_income_ratio'] = df['monthly_contribution'] / (df['annual_income'] / 12 + 1e-5)
df['volatility_risk_interaction'] = df['risk_appetite'] * df['market_volatility_index']
df['investment_density'] = df['portfolio_value'] / (df['investment_horizon_years'] + 1e-5)

# Define Features & Targets
features = [
    'risk_appetite', 'portfolio_value', 'monthly_contribution', 'equity_allocation_pct','debt_to_income_ratio',
    'fixed_income_allocation_pct', 'market_volatility_index','annual_income','macroeconomic_score','preferred_asset_classes',
    'employment_status_Retired','employment_status_Salaried','employment_status_Self-Employed','employment_status_Unemployed',
    'investment_horizon_years', 'savings_rate', 'effective_savings','financial_knowledge_score','net_worth',
    'contribution_to_income_ratio', 'volatility_risk_interaction', 'investment_density'
]
X = df[features]
y = df[['forecasted_value_year_1', 'forecasted_value_year_2', 'forecasted_value_year_3']]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
base_model = XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train_scaled, y_train)

# Make Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Performance
for i, year in enumerate(['Year 1', 'Year 2', 'Year 3']):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\n📊Performance for {year}:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

# Predict Future Portfolio Values
future_values = model.predict(X_test_scaled[:5])
print("\n Predicted Portfolio Values for Next 3 Years:(Example for 5 clients)\n", future_values)

# Feature Importance Plot
importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=X.columns, palette="viridis")
plt.title("Average Feature Importance (XGBoost across 3 years)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Predict for a specific client
user_input_id = input("Enter Client UUID (e.g., 96c4c0a3-bb3f-4ac1-81ad-0850cd29911f): ")
client_row = df[df['client_id'] == user_input_id]

if client_row.empty:
    print(" Client ID not found.")
else:
    client_features = client_row[features]
    client_scaled = scaler.transform(client_features)
    client_prediction = model.predict(client_scaled)
    print(f"\n Forecasted Portfolio Values for {user_input_id}:")
    print(f"Year 1: ₹{client_prediction[0][0]:,.2f}")
    print(f"Year 2: ₹{client_prediction[0][1]:,.2f}")
    print(f"Year 3: ₹{client_prediction[0][2]:,.2f}")

# Load macro scenarios
macro_df = pd.read_csv("C:/Users/SHRUTI-NIDHI/OneDrive/Desktop/macro_scenarios.csv")

# Macro Scenario Adjustment Function
def apply_macro_scenario(scenario_id, client_uuid):
    scenario = macro_df[macro_df['scenario_id'] == scenario_id].iloc[0]
    adjusted_X = X_test.copy()
    adjusted_client_ids = df.loc[X_test.index, 'client_id'].reset_index(drop=True)

    portfolio_value_adjustment = scenario['equity_impact'] + scenario['fixed_income_impact']
    adjusted_X['portfolio_value'] *= (1 + portfolio_value_adjustment)
    adjusted_X['monthly_contribution'] *= (1 + scenario['equity_impact'])
    adjusted_X['macroeconomic_score'] += scenario['macroeconomic_score_adjustment'] + scenario['sentiment_index_adjustment']
    adjusted_X['savings_rate'] *= (1 - scenario['inflation_spike'])
    adjusted_X['savings_rate'] = adjusted_X['savings_rate'].clip(lower=0)
    adjusted_X['fixed_income_allocation_pct'] *= (1 + scenario['interest_rate_change'])
    adjusted_X['market_volatility_index'] *= (1 + scenario['market_volatility_shock'])

    adjusted_X['effective_savings'] = adjusted_X['portfolio_value'] * adjusted_X['savings_rate']
    adjusted_X['contribution_to_income_ratio'] = adjusted_X['monthly_contribution'] / (adjusted_X['annual_income'] / 12 + 1e-5)
    adjusted_X['volatility_risk_interaction'] = adjusted_X['risk_appetite'] * adjusted_X['market_volatility_index']
    adjusted_X['investment_density'] = adjusted_X['portfolio_value'] / (adjusted_X['investment_horizon_years'] + 1e-5)

    adjusted_X_scaled = scaler.transform(adjusted_X)
    adjusted_pred = model.predict(adjusted_X_scaled)

    try:
        client_idx = adjusted_client_ids[adjusted_client_ids == client_uuid].index[0]
    except IndexError:
        print(" Client UUID not found in test data.")
        return

    comparison_df = pd.DataFrame({
        'Original': client_prediction[0],
        'Adjusted': adjusted_pred[client_idx]
    }, index=["Year 1", "Year 2", "Year 3"])

    print(f"\nComparison of Original vs Adjusted Predictions for Scenario {scenario_id} and Client {client_uuid}:")
    print(comparison_df.round(2))

    plt.figure(figsize=(6, 4))
    plt.plot(comparison_df['Original'], label="Original", marker='o')
    plt.plot(comparison_df['Adjusted'], label="Adjusted", marker='x')
    plt.title(f"Scenario Impact: {scenario_id} - Client {client_uuid[:6]}...")
    plt.xlabel("Year")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Inputs for Scenario and Client UUID
scenario_input = input("Enter Scenario ID (e.g., SCN_01): ")
client_id_input = input("Enter Client UUID (e.g., 96c4c0a3-bb3f-4ac1-81ad-0850cd29911f): ")
apply_macro_scenario(scenario_input, client_id_input)