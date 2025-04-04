import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("encoded.csv")


features = [
    'risk_appetite', 'investment_goals_Education', 'investment_goals_Home Purchase', 
    'investment_goals_Retirement', 'investment_goals_Wealth Accumulation', 
    'financial_knowledge_score', 'portfolio_value'
]
target = 'recommended_strategy'

X = df[features].copy()  
y = df[target]

categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])


y = LabelEncoder().fit_transform(y) 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

 #MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)

print("\n✅ Multi-Layer Perceptron Model Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
