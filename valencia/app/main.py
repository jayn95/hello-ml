from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from dotenv import load_dotenv
from google import genai  # Import Google GenAI library
from pydantic import BaseModel, ValidationError
from typing import List

class ResponseSchema(BaseModel):
    Description: str
    Causes: List[str]
    RiskFactors: List[str]
    Prognosis: str
    Treatments: List[str]


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the TFLite model
model_path = os.path.join('model', 'model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
labels_path = os.path.join('model', 'labels.json')
with open(labels_path, 'r') as f:
    labels = json.load(f)

# Configure Google GenAI
genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
if not genai_api_key:
    raise ValueError("Google GenAI API key not found. Please set it in the .env file.")
client = genai.Client(api_key=genai_api_key)


def fetch_info_from_genai(skin_cancer_type):
    """
    Fetch detailed information about the classified skin cancer type using Google GenAI.
    """
    try:
        # Define separate prompts for each category
        prompts = {
            "Description": f"Provide a concise description (5-10 sentences) of {skin_cancer_type} without any labels just the description and not with description:. Remove disclaimer and any other text.",
            "Causes": f"List the main causes of {skin_cancer_type} in a structured format and don't mention the name of the cancer in the output and just pure output don't answer with anything else. Separate the causes with commas. remove disclaimer and any other text.",
            "Risk Factors": f"List the main risk factors of {skin_cancer_type} in a structured format and don't mention the name of the cancer in the output and just pure output don't answer with anything else. Separate the risk factors with commas. remove disclaimer and any other text.",
            "Prognosis": f"Provide the average prognosis for {skin_cancer_type} and prognosis only not anything else with percentage of survivability. Don't mention the name of the cancer in the output just pure answer and no labels. remove disclaimer and any other text.",
            "Treatments": f"List the available treatments for {skin_cancer_type} without descriptions and just pure answer to the question without anything else. Separate the treatments with new lines. remove disclaimer and any other text."
        }

        # Fetch information for each category
        details = {}
        for category, prompt in prompts.items():
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "text/plain"
                }
            )
            # Parse and clean the response
            raw_text = response.text.strip()
            cleaned_text = (
                raw_text.replace("Description:", "")
                        .replace("description:", "")
                        .replace("Causes:", "")
                        .replace("causes:", "")
                        .replace("Risk Factors:", "")
                        .replace("risk factors:", "")
                        .replace("Prognosis:", "")
                        .replace("prognosis:", "")
                        .replace("Treatments:", "")
                        .replace("treatments:", "")
                        .replace("Basal Cell Carcinoma:", "")
                        .replace("{", "")
                        .replace("}", "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace('"', "")  # Remove all quotes
                        .strip()
            )
            # For Treatments, split into a list of items
            if category == "Treatments":
                details[category] = [treatment.strip() for treatment in cleaned_text.split("\n") if treatment.strip()]
            else:
                details[category] = cleaned_text

        # Validate and structure the response using the schema
        structured_response = ResponseSchema(
            Description=details.get("Description", "Information not available."),
            Causes=[cause.strip() for cause in details.get("Causes", "Information not available.").split(",")],
            RiskFactors=[risk.strip() for risk in details.get("Risk Factors", "Information not available.").split(",")],
            Prognosis=details.get("Prognosis", "Information not available."),
            Treatments=details.get("Treatments", [])
        )

        return structured_response.dict()
    except ValidationError as ve:
        print(f"Validation error: {ve}")
        return {
            "Description": "Information not available.",
            "Causes": [],
            "RiskFactors": [],
            "Prognosis": "Information not available.",
            "Treatments": []
        }
    except Exception as e:
        print(f"Error fetching data from GenAI: {e}")
        return {
            "Description": "Information not available.",
            "Causes": [],
            "RiskFactors": [],
            "Prognosis": "Information not available.",
            "Treatments": []
        }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None, image_url=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        # Save the uploaded image for display
        upload_folder = os.path.join(app.static_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # Ensure the directory exists
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)

        # Process the image
        image = Image.open(image_path)
        image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process the output data
        result_index = np.argmax(output_data[0])
        result_label = labels[result_index]
        result_confidence = output_data[0][result_index] * 100  # Convert to percentage

        # Extract only the name from the classification result
        result_name = result_label.get("name", "Unknown")

        # Fetch additional information from Google GenAI
        additional_info = fetch_info_from_genai(result_name)

        # Pass the relative path to the template
        relative_image_path = f"uploads/{file.filename}"

        return render_template(
            'index.html',
            result={
                "label": result_name,
                "confidence": f"{result_confidence:.2f}%",  # Format as percentage
                "details": additional_info,
            },
            image_url=url_for('static', filename=relative_image_path)  # Use url_for for static files
        )

    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)