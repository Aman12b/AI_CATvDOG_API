import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify
import tempfile
import os

app = Flask(__name__)

# Load your model (assuming the model file is present in the same directory)
#test
model = load_model('cat_vs_dog_classifier.h5')

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/catordog', methods=['POST'])
def cat_or_dog():
    try:

        img_file = request.files['file']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            img_file.save(temp.name)
            temp_path = temp.name


        img_array = load_and_preprocess_image(temp_path)

        prediction = model.predict(img_array)

        os.remove(temp_path)
        result = "dog" if prediction[0] > 0.5 else "cat"
        return jsonify({"result": result, "confidence": float(prediction[0]) if prediction[0] > 0.5 else 1-float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
