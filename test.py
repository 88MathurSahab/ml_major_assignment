# test.py
import joblib
from sklearn.metrics import accuracy_score

def test_model():
    # Load model
    try:
        clf = joblib.load('savedmodel.pth')
        # Load test data (saved during training for consistency)
        X_test, y_test = joblib.load('test_data.pkl')
    except FileNotFoundError:
        print("Error: Model or test data not found. Run train.py first.")
        return

    # Predict
    y_pred = clf.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test_model()