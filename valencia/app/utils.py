def load_model(model_path):
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=model_path)

def preprocess_image(image, target_size):
    from PIL import Image
    import numpy as np

    image = Image.open(image)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def predict(model, processed_image):
    model.allocate_tensors()
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], processed_image)
    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])
    return output_data