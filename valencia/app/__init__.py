from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import os
from utils import load_model, predict

app = Flask(__name__)

# Load the TensorFlow Lite model
model = load_model(os.path.join('model', 'model.tflite'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Process the image and get predictions
        result = predict(file, model)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)