import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = os.path.join("static", "uploads")  # Fix: use os.path.join for Windows
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Auto-create the uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("models/animal_model.keras")

# Class names (must match folder names EXACTLY)
class_names = ['butterfly', 'cat', 'cow', 'dog',
               'elephant', 'horse', 'sheep', 'squirrel']


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html")

        file = request.files['file']

        if file.filename == "":
            return render_template("index.html")

        # ✅ Use forward slashes for the web-accessible path
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename).replace("\\", "/")
        file.save(img_path)

        processed_image = prepare_image(img_path)
        preds = model.predict(processed_image)

        predicted_class = class_names[np.argmax(preds)]
        confidence = round(np.max(preds) * 100, 2)

        prediction = predicted_class

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)