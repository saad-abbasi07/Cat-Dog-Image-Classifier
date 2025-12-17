import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model("cat_dog_model.keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "predict": "POST /predict with image file"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files["file"]
        
        # Validate image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        
        # Preprocess
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Format result
        result = "Dog" if prediction > 0.5 else "Cat"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return jsonify({
            "prediction": result,
            "confidence": round(confidence * 100, 2),
            "raw_score": float(prediction),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "API is working"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)