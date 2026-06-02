import joblib
from sklearn.datasets import fetch_olivetti_faces

# Load model
model = joblib.load("face_model.pkl")

# Load dataset
data = fetch_olivetti_faces()

X = data.data
y = data.target

# Predict first image
prediction = model.predict([X[0]])

print("Predicted Person:", prediction[0])
print("Actual Person:", y[0])