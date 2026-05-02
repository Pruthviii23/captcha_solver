import cv2
import json
import numpy as np
import tensorflow as tf

IMG_WIDTH = 160
IMG_HEIGHT = 75

# ==========================
# LOAD MODEL + VOCAB
# ==========================

model = tf.keras.models.load_model("captcha_model.keras")

with open("char_vocab.json", "r") as f:
    characters = json.load(f)

num_to_char = {i:c for i,c in enumerate(characters)}

# ==========================
# PREPROCESS
# ==========================

def preprocess(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0

    return img.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

# ==========================
# DECODE (🔥 beam search)
# ==========================

def decode(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=10   # 🔥 important
    )[0][0]

    results = []
    for seq in decoded:
        text = ""
        for i in seq:
            if i == -1:
                continue
            text += num_to_char[int(i)]
        results.append(text)

    return results

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