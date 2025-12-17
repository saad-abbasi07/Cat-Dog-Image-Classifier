import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model (make sure path is correct)
model = tf.keras.models.load_model("cat_dog_model.keras")

@app.route("/")
def home():
    return "Cat Dog Classifier is running! Use /predict endpoint."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0][0]
        result = "Dog" if prediction > 0.5 else "Cat"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return {
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "raw_score": float(prediction)
        }
    
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)