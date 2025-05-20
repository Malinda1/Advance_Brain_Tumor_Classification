import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Improved import handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    print("✅ TensorFlow backend loaded")
except ImportError as e:
    print(f"❌ TensorFlow import error: {e}")
    try:
        from keras.models import load_model
        from keras.preprocessing import image
        print("✅ Standalone Keras loaded")
    except ImportError as e:
        print(f"❌ Keras import error: {e}")
        raise ImportError("Neither TensorFlow nor standalone Keras could be imported")

app = Flask(__name__)
CORS(app)

app.secret_key = 'your_secret_key_here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'best_model.h5'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model with improved error handling
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, webp'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if model is None:
            return jsonify({'error': 'Model failed to load. Contact administrator.'}), 500

        processed_img = preprocess_image(filepath)
        if processed_img is None:
            return jsonify({'error': 'Failed to process image'}), 400

        prediction = model.predict(processed_img)
        confidence = float(prediction[0][0])
        result = "No Brain Tumor" if confidence > 0.5 else "Brain Tumor Detected"
        confidence_pct = confidence if result.startswith("No") else (1 - confidence)

        return jsonify({
            'prediction': result,
            'confidence': round(confidence_pct * 100, 2),
            'filename': filename
        })
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)