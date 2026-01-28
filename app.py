# -------------------- IMPORTS --------------------
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# -------------------- LOAD MODEL --------------------
MODEL_PATH = "models/dog_breed_inceptionv3.keras"
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'beagle', 'bulldog', 'dalmatian', 'german-shepherd',
    'husky', 'labrador-retriever', 'poodle', 'rottweiler'
]

def predict_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100

    return class_labels[idx], confidence

# -------------------- ROUTES --------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", message="No file selected")

        if allowed_file(file.filename):
            os.makedirs("static", exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join("static", filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            predicted_label, confidence = predict_image(img)

            return render_template(
                "index.html",
                image_path=filepath,
                predicted_label=predicted_label,
                confidence=confidence
            )

    return render_template("index.html")

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
