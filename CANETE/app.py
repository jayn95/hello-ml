import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs except errors

app = Flask(__name__)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle API key - try environment variable first, then .env file
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    # Try loading from .env as fallback
    load_dotenv()
    API_KEY = os.environ.get('GEMINI_API_KEY')
    
if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables or .env file")
    raise ValueError("GEMINI_API_KEY not found. Please set this environment variable in Render dashboard.")

genai.configure(api_key=API_KEY)

# Create necessary directories
for directory in ['static/uploads', 'static/images']:
    os.makedirs(directory, exist_ok=True)

# Define the base directory for the model and label files
# For Render, ensure these files are in your repository
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skin_cancer_model')
logger.info(f"Base directory: {base_dir}")

# Create the model directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Check if model directory has contents
try:
    dir_contents = os.listdir(base_dir)
    logger.info(f"Contents of {base_dir}: {dir_contents}")
except Exception as e:
    logger.warning(f"Couldn't list directory contents: {str(e)}")

# Load the TFLite model
model_path = os.path.join(base_dir, 'model.tflite')
logger.info(f"Looking for model at: {model_path}")

if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(
        f'Model file not found at {model_path}. '
        'Please ensure the "skin_cancer_model" directory and its contents are included in your deployment.'
    )

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.info("TF Lite model loaded successfully")
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Model input shape: {input_details[0]['shape']}")
except Exception as e:
    logger.error(f"Error loading TF Lite model: {str(e)}")
    raise

# Load labels from labels.txt
labels_path = os.path.join(base_dir, 'labels.txt')
if not os.path.exists(labels_path):
    logger.error(f"Labels file not found at {labels_path}")
    raise FileNotFoundError(f'Labels file not found at {labels_path}. Please check your deployment.')

with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')
    logger.info(f"Loaded {len(labels)} labels")

# Load label names from labels.json
labels_json_path = os.path.join(base_dir, 'labels.json')
if not os.path.exists(labels_json_path):
    logger.error(f"Labels JSON file not found at {labels_json_path}")
    raise FileNotFoundError(f'Labels JSON file not found at {labels_json_path}. Please check your deployment.')

with open(labels_json_path, 'r') as f:
    try:
        label_data = json.load(f)
        label_names = {item['label']: item['name'] for item in label_data}
        logger.info(f"Loaded {len(label_names)} label names")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing labels.json: {str(e)}")
        raise

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        logger.info(f"Preprocessing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to match model's expected input size
        image = np.array(image) / 255.0   # Normalize pixel values
        image = np.expand_dims(image, axis=0).astype(np.float32)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Error preprocessing image: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Prediction request received")
        file_path = None
        
        # Handle example images or uploaded files
        if 'example' in request.form:
            example_image = request.form['example']
            logger.info(f"Example image requested: {example_image}")
            
            # Try different possible paths for the example image
            possible_paths = [
                os.path.join('static', 'images', example_image),  # Relative path
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images', example_image),  # Absolute path
                os.path.abspath(os.path.join('static', 'images', example_image))  # Another absolute path format
            ]
            
            # Try to find the file in any of the possible locations
            for path in possible_paths:
                logger.info(f"Checking path: {path}")
                if os.path.exists(path):
                    file_path = path
                    logger.info(f"Found example image at: {file_path}")
                    break
            
            # If we still couldn't find the file
            if file_path is None or not os.path.exists(file_path):
                logger.error(f"Example image not found in any of the checked paths")
                
                # Debug directory contents
                try:
                    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
                    if os.path.exists(static_dir):
                        logger.info(f"Static directory exists. Contents: {os.listdir(static_dir)}")
                        images_dir = os.path.join(static_dir, 'images')
                        if os.path.exists(images_dir):
                            logger.info(f"Images directory exists. Contents: {os.listdir(images_dir)}")
                        else:
                            logger.error(f"Images directory doesn't exist at {images_dir}")
                    else:
                        logger.error(f"Static directory doesn't exist at {static_dir}")
                except Exception as e:
                    logger.error(f"Error checking directories: {str(e)}")
                
                return jsonify({
                    'error': f'Example image {example_image} not found. Please upload your own image instead.'
                }), 404
        else:
            if 'file' not in request.files:
                logger.warning("No file uploaded")
                return jsonify({'error': 'No file uploaded.'}), 400

            file = request.files['file']
            if file.filename == '':
                logger.warning("Empty filename")
                return jsonify({'error': 'No selected file.'}), 400

            if not allowed_file(file.filename):
                logger.warning(f"Invalid file type: {file.filename}")
                return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed.'}), 400

            # Ensure the uploads directory exists
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            file_path = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(file_path)
            logger.info(f"File saved to {file_path}")

        # Verify file exists and is readable before proceeding
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return jsonify({'error': 'File not found after upload.'}), 500
            
        try:
            with open(file_path, 'rb') as f:
                logger.info(f"File is readable and has size: {os.path.getsize(file_path)} bytes")
        except Exception as e:
            logger.error(f"File exists but cannot be read: {str(e)}")
            return jsonify({'error': 'File cannot be read.'}), 500

        # Preprocess the image
        try:
            input_data = preprocess_image(file_path)
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        # Perform inference
        try:
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            logger.info(f"Prediction successful, raw output: {predictions[:3]}...")  # Log first 3 values
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return jsonify({'error': f'Error during model inference: {str(e)}'}), 500

        # Get the predicted label and confidence
        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        
        # Check if predicted label exists in label names
        if predicted_label not in label_names:
            logger.error(f"Label {predicted_label} not found in label_names dictionary")
            return jsonify({'error': f'Unknown label: {predicted_label}'}), 500
            
        predicted_name = label_names[predicted_label]
        confidence = float(predictions[predicted_index])
        logger.info(f"Prediction: {predicted_name} ({predicted_label}) with confidence {confidence:.4f}")

        # Clean up uploaded file if needed - but not example images
        if 'file' in request.files and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove file {file_path}: {str(e)}")

        # Generate treatment and description using Google Generative AI
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f'Provide a detailed description and recommended treatment for {predicted_name} in 8 sentences.'
            logger.info(f"Sending prompt to Gemini: {prompt}")
            
            response = model.generate_content(prompt)
            description_and_treatment = response.text
            logger.info("Generated treatment description successfully")
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            description_and_treatment = f"Could not generate description for {predicted_name}. Please consult a healthcare professional for diagnosis and treatment."

        return jsonify({
            'label': predicted_label,
            'name': predicted_name,
            'confidence': confidence,
            'description_and_treatment': description_and_treatment
        })
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)