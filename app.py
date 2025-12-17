from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# load model (make sure path is correct)
model = tf.keras.models.load_model("cat_dog_model.keras")

@app.route("/")
def home():
    return "Cat Dog Classifier is running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    result = "Dog" if prediction > 0.5 else "Cat"

    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
