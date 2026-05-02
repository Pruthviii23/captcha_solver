import cv2
import json
import numpy as np
import tensorflow as tf

IMG_WIDTH = 160
IMG_HEIGHT = 75

# ==========================
# LOAD
# ==========================

model = tf.keras.models.load_model("captcha_model.keras")

with open("char_vocab.json", "r") as f:
    characters = json.load(f)

num_to_char = {i+1:c for i,c in enumerate(characters)}

# ==========================
# PREPROCESS
# ==========================

def preprocess(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0

    return img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

# ==========================
# DECODE
# ==========================

def decode(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=10
    )[0][0]

    text = ""

    for i in decoded[0]:
        if i <= 0:
            continue
        text += num_to_char.get(int(i), "")

    return text

# ==========================
# TEST
# ==========================

img1 = preprocess("captcha_dataset/images/captcha_03051.png")
img2 = preprocess("captcha_dataset/images/captcha_03052.png")
img3 = preprocess("captcha_dataset/images/captcha_03053.png")
img4 = preprocess("captcha_dataset/images/captcha_03054.png")
img5 = preprocess("captcha_dataset/images/captcha_03055.png")

pred1 = model.predict(img1)
pred2 = model.predict(img2)
pred3 = model.predict(img3)
pred4 = model.predict(img4)
pred5 = model.predict(img5)

print("Prediction:", decode(pred1))
print("Prediction:", decode(pred2))
print("Prediction:", decode(pred3))
print("Prediction:", decode(pred4))
print("Prediction:", decode(pred5))