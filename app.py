from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/oral_model.h5")

# Base CNN (MobileNetV2)
base_model = model.layers[0]

# Last conv layer
last_conv_layer = base_model.get_layer("Conv_1")

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# GRAD-CAM
# =========================
def generate_gradcam(img_array):
    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)

        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(x)

        predictions = x
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((224,224,3), dtype=np.uint8)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (224,224))

    # 🔥 Improve heatmap quality
    heatmap = cv2.GaussianBlur(heatmap, (15,15), 0)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB')

        processed = preprocess(img)

        # Prediction
        pred = model.predict(processed)[0][0]

        prob_normal = float(pred)
        prob_cancer = float(1 - pred)

        # Risk classification
        if prob_cancer > 0.7:
            risk = "High Risk"
        elif prob_cancer > 0.4:
            risk = "Suspicious"
        else:
            risk = "Normal"

        confidence = float(round(prob_cancer * 100, 2))

        # Grad-CAM
        heatmap = generate_gradcam(processed)
        original = np.array(img.resize((224,224)))

        # 🔥 Better overlay
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        _, buffer = cv2.imencode('.png', superimposed.astype('uint8'))
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        # Message
        if risk == "High Risk":
            msg = "Strong signs detected. Consult a specialist immediately."
        elif risk == "Suspicious":
            msg = "Some abnormal patterns detected. Medical consultation advised."
        else:
            msg = "No major abnormalities detected. Maintain regular checkups."

        return jsonify({
            "risk_level": risk,
            "confidence": confidence,
            "message": msg,
            "heatmap": heatmap_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)