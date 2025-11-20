# app.py
from flask import Flask, request, render_template_string
import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import io

app = Flask(__name__)
MODEL_PATH = 'savedmodel.pth'

# Load the model upon startup
try:
    model = joblib.load(MODEL_PATH)
    # Load Olivetti faces to get target names (person IDs)
    data = fetch_olivetti_faces()
    TARGET_NAMES = [f"Person {i}" for i in data.target_names]
except Exception as e:
    print(f"Error loading model or data: {e}")
    model = None
    TARGET_NAMES = ["Error"]

# Simple HTML Template for upload and display
HTML_TEMPLATE = """
<!doctype html>
<title>Olivetti Face Predictor</title>
<h1>Upload a Face Image (Requires 64x64 Grayscale)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h2>Prediction Result: {{ prediction }}</h2>
{% endif %}
"""

def preprocess_image(file_data):
    # Process image to match Olivetti faces input (64x64, grayscale, normalized)
    img = Image.open(io.BytesIO(file_data)).convert('L') # Convert to grayscale
    img = img.resize((64, 64)) # Resize to match Olivetti
    img_array = np.array(img) / 255.0 # Normalize pixel values (0-1)
    img_flat = img_array.reshape(1, -1) # Flatten to (1, 4096)
    return img_flat

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file and model:
            try:
                # Preprocess the image for the Decision Tree Model
                X_input = preprocess_image(file.read())
                
                # Predict
                pred_class = model.predict(X_input)[0]
                prediction = TARGET_NAMES[pred_class]
                
            except Exception as e:
                prediction = f"Prediction Error: {e}"
        elif not model:
            prediction = "Model failed to load on server startup."
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    # Flask runs on 0.0.0.0 for Docker/K8s accessibility
    app.run(host='0.0.0.0', port=5000)