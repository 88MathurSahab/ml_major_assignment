# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

def train_model():
    # Load dataset
    print("Loading Olivetti Faces dataset...")
    data = fetch_olivetti_faces()
    X, y = data.data, data.target

    # Split data: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    print("Training Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, 'savedmodel.pth')
    print("Model saved as savedmodel.pth")

    # Save test data for testing script (optional but helpful for consistency)
    joblib.dump((X_test, y_test), 'test_data.pkl')

if __name__ == "__main__":
    train_model()