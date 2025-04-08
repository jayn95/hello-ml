from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

# Configure Generative AI API
genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)

# Base directory of the current script
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_image(img_path, input_shape):
    """Load and preprocess an image for the model."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def load_labels(label_path):
    """Load labels from a text file."""
    with open(label_path, 'r') as file:
        labels = file.readlines()
    return {str(index): label.strip() for index, label in enumerate(labels)}

def load_labels_json(label_path):
    """Load detailed labels from a JSON file."""
    with open(label_path, 'r') as file:
        labels = json.load(file)
    return {item['label']: item['name'] for item in labels}

def fetch_disease_info(disease_name):
    """
    Generate a detailed description of the disease using Gemini's generative AI model.
    Fallback to static information if the model fails.
    """
    # Static fallback information
    static_info = {
        "Melanoma": "Melanoma is a type of skin cancer that develops in melanocytes. It is often caused by excessive exposure to ultraviolet (UV) radiation. Early detection and treatment are critical.",
        "Basal Cell Carcinoma": "Basal Cell Carcinoma is a common type of skin cancer that arises from basal cells. It is usually caused by prolonged sun exposure and is highly treatable.",
        "Benign Keratosis": "Benign Keratosis is a non-cancerous skin condition that often appears as rough, scaly patches on the skin. It is usually harmless and does not require treatment.",
        # Add more diseases as needed
    }

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Enhanced prompt for better AI response
        prompt = (f"Provide a concise and informative description of {disease_name}. "
                  f"Include its symptoms, possible causes, diagnosis methods, "
                  f"and recommended treatments or prevention methods.")
        
        response = model.generate_content(prompt)
        
        # Ensure response is valid before returning text
        if response and hasattr(response, "text"):
            return response.text
        else:
            # Fallback to static information
            return static_info.get(disease_name, "No additional information available.")
    
    except Exception as e:
        print(f"Error generating disease information: {e}")
        # Fallback to static information
        return static_info.get(disease_name, "Unable to generate information about the disease at this time.")
    
@app.route('/')
def index():
    """Render the homepage with an upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make predictions."""
    if 'file' not in request.files:
        return render_template('error.html', error="No file uploaded"), 400

    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', error="No file selected"), 400

    # Save the uploaded file
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Load the TFLite model and allocate tensors
    model_path = r"model/model.tflite"


    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Extract input shape dynamically
    input_shape = input_details[0]['shape']

    # Preprocess the image
    img_array = load_and_preprocess_image(img_path, input_shape)

    # Prepare the input data
    try:
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = np.squeeze(output_data)
    except Exception as e:
        return render_template('error.html', error=f"Error during inference: {e}"), 500

    # Load labels from labels.txt and labels.json
    labels_txt = load_labels("hello-ml/navarra/model/labels.txt")
    labels_json = load_labels_json("hello-ml/navarra/model/labels.json")

    # Prepare the predictions
    results = []
    for i, score in enumerate(predictions):
        label_key = labels_txt.get(str(i), f"Unknown label {i}")
        label_name = labels_json.get(label_key, "Unknown")  # Map label to name from labels.json
        results.append({
            'label': label_name,
            'score': float(score)
        })

    # Determine the disease with the highest prediction score
    highest_prediction = max(results, key=lambda x: x['score'])
    classification = (f"The image is most likely classified as {highest_prediction['label']} "
                      f"with a confidence score of {highest_prediction['score']:.2f}.")

    # Generate detailed information about the disease using Gemini AI
    disease_info = fetch_disease_info(highest_prediction['label'])

    # Render the results in an HTML template
    return render_template('results.html', results=results, classification=classification, disease_info=disease_info)

if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
