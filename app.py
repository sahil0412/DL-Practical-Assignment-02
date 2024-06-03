import os
import warnings
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from collections import OrderedDict

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the best trained model
model_path = 'best_model.h5'
model = load_model(model_path, compile=False)

# Serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image'].read()
    
    # Preprocess the image
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = np.expand_dims(image, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Return the prediction as a JSON response
    return jsonify({'predicted_class': int(predicted_class)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
