import cv2
import json
import numpy as np
import tensorflow as tf

IMG_WIDTH = 160
IMG_HEIGHT = 75

model = tf.keras.models.load_model("captcha_model_multiclass.keras")

with open("char_vocab.json", "r") as f:
    characters = json.load(f)

def preprocess(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0

    return img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

img = preprocess("captcha_dataset/images/captcha_03052.png")

preds = model.predict(img)

text = ""
for p in preds:
    idx = np.argmax(p)
    text += characters[idx]

print("Prediction:", text)