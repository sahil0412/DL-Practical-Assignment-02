import os
import warnings
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Define the static and upload directories
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the best trained model
model_path = 'best_model.h5'
model = load_model(model_path, compile=False)

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # image = request.files['image'].read()
        
        # # Preprocess the image
        # image = tf.image.decode_image(image, channels=3)
        # image = tf.image.resize(image, [32, 32])
        # image = np.expand_dims(image, axis=0) / 255.0
        
        # # Make prediction
        # predictions = model.predict(image)
        # predicted_class = np.argmax(predictions, axis=1)[0]
        
        
        
        file = request.files['file']

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and preprocess the image
            file_img = cv2.imread(filepath)
            face = cv2.resize(file_img, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Make prediction using your deep learning model
            prediction = model.predict(face)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Pass prediction and image file path to result template
            return render_template('result.html', prediction=predicted_class, image_file=filename)

if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
