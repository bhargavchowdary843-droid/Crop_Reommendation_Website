import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

print(df.head())

# Features
X = df.drop("label", axis=1)

# Target (keep crop names as strings)
y = df["label"]

print("Training model...")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("Saving model...")

# Save model
joblib.dump(model, "crop_model.pkl")

print("SUCCESS!")
print("Sample prediction:")
print(model.predict([X.iloc[0]])[0])