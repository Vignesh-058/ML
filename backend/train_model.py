# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
data = pd.read_csv("student_scores.csv")  # Make sure this file is in the same folder
X = data[["hours"]]
y = data["marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
