import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("data/data.csv")
X = df[['age', 'salary']]
y = df['purchased']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "data/model.joblib")

print("✅ Model trained and saved")

