import os, json

from flask import Flask, render_template, request, jsonify
from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np


def create_app(allowed_extensions, upload_folder='uploads', model_path='model.tflite'):
  app = Flask(__name__)

  # Configure upload folder
  UPLOAD_FOLDER = upload_folder
  ALLOWED_EXTENSIONS = allowed_extensions
  MODEL_PATH = model_path
  IMG_SIZE = 224
  LABEL_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  # Initialize TFLite interpreter
  interpreter = tflite.Interpreter(model_path=os.path.join(os.getcwd(), MODEL_PATH))
  interpreter.allocate_tensors()

  # Get input and output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  os.makedirs(UPLOAD_FOLDER, exist_ok=True)

  def preprocess_image(image_path):
      img = Image.open(image_path)
      img = img.resize((IMG_SIZE, IMG_SIZE))
      img_array = np.array(img, dtype=np.float32)
      img_array = img_array / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      return img_array

  def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

  @app.route('/')
  def home():
    return render_template('index.html')

  @app.route('/predict', methods=['POST'])
  def predict():
    if 'file' not in request.files:
      return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
      return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
      
      try:
        # Save and process image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        processed_image = preprocess_image(filename)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Run inference
        interpreter.invoke()

        # Get prediction results
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = LABEL_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        # Clean up
        os.remove(filename)

        labels = json.loads(open(os.path.join(os.getcwd(), 'labels.json'), 'r').read())
        name = [obj['name'] for obj in labels if obj['label'] == predicted_class]

        return jsonify({
            'prediction': predicted_class,
            'name': name,
            'confidence': f'{confidence:.2%}'
        })

      except Exception as e:
              return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

  return app


app = create_app({'png', 'jpg', 'jpeg'})
