#and this is main.py

#main.py
#if it needs any correction?
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import os

app = Flask(__name__)

# ===============================
# ✅ Load SavedModel (FINAL)
# ===============================
#model = tf.keras.models.load_model('models/final_model')
import tensorflow as tf

loaded = tf.saved_model.load('models/final_model/content/drive/MyDrive/final_model')
infer = loaded.signatures['serving_default']
print(infer.structured_input_signature)
# Get inference function
#infer = loaded.signatures['serving_default']
# ===============================
# Classes (MUST match training)
# ===============================

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ===============================
# Image settings
# ===============================
IMG_SIZE = (224, 224)

# ===============================
# Preprocessing (IMPORTANT)
# ===============================
#def preprocess(img):
    #img = img / 255.0   # change if needed
    #return img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(img):
    return preprocess_input(img)
# ===============================
# Upload folder
# ===============================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Prediction function
# ===============================
def predict_tumor(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = np.array(img).astype('float32')

    img = preprocess(img)
    img = np.expand_dims(img, axis=0)

    # 🔥 universal call (no key needed)
    preds = infer(tf.constant(img))
    preds = list(preds.values())[0].numpy()

    pred_index = np.argmax(preds)
    confidence = float(np.max(preds))

    label = classes[pred_index]

    if label == 'notumor':
        return "No Tumor Detected", confidence
    else:
        return f"Tumor: {label}", confidence
# ===============================
# Routes
# ===============================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            prediction, confidence = predict_tumor(filepath)

            return render_template(
                'index.html',
                prediction=prediction,
                confidence=round(confidence * 100, 2),
                filename=file.filename
            )

    return render_template('index.html')

# ===============================
# Run app
# ===============================
if __name__ == '__main__':
    app.run(debug=True)