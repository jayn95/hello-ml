# Skin Cancer Classifier Flask App

This project is a Flask-based web application that utilizes a TensorFlow Lite model to classify skin cancer types from uploaded images. It also provides detailed information about the classified type using Google GenAI. The application is designed to be simple, user-friendly, and visually appealing, making it suitable for medical professionals and researchers.

Access Online for Demo: https://skin-cancer-classifier-m1pd.onrender.com


---

## Features

1. **Image Upload and Classification**:
   - Users can upload an image of a skin lesion.
   - The application processes the image using a TensorFlow Lite model to classify the type of skin cancer.

2. **Detailed Analysis**:
   - After classification, the app fetches additional information about the classified type using Google GenAI, including:
     - Description
     - Causes
     - Risk Factors
     - Prognosis
     - Treatments

3. **Interactive UI**:
   - A clean and responsive user interface built with Tailwind CSS.
   - The layout includes:
     - An upload section for images.
     - A preview of the uploaded image.
     - A structured display of classification results and additional details.

4. **Cloud and Local Deployment**:
   - The app can run locally or be deployed on cloud platforms like Render.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd skin-cancer-classifier
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables
- Create a `.env` file in the root directory.
- Add your Google GenAI API key:
  ```
  GOOGLE_GENAI_API_KEY=your_api_key_here
  ```

### 5. Run the Application
```bash
python app/main.py
```

### 6. Access the Application
- Open your web browser and navigate to: `http://127.0.0.1:5000`

---

## Usage

1. **Upload an Image**:
   - On the main page, upload an image of a skin lesion.

2. **View Results**:
   - The app will classify the image and display:
     - The type of skin cancer.
     - Confidence score.
     - Detailed information fetched from Google GenAI.

3. **Analyze the Details**:
   - Review the structured details, including causes, risk factors, prognosis, and treatments.

---

## Technologies Used

- **Flask**: A lightweight web framework for Python.
- **TensorFlow Lite**: A lightweight solution for running machine learning models.
- **Google GenAI**: For fetching detailed information about the classified type.
- **Tailwind CSS**: A utility-first CSS framework for responsive and modern UI design.
- **Pillow**: For image processing.
- **Pydantic**: For data validation and structuring.

---

## Example Workflow

1. **Upload**: A user uploads an image of a skin lesion.
2. **Classification**: The TensorFlow Lite model classifies the image.
3. **Details**: The app fetches additional information about the classified type using Google GenAI.
4. **Results**: The app displays the results in a structured and user-friendly format.

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- TensorFlow Lite for providing a lightweight machine learning solution.
- Google GenAI for generating detailed information.
- Tailwind CSS for the modern and responsive UI design.
