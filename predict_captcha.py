import cv2
import json
import numpy as np
import tensorflow as tf

# ==========================
# SETTINGS
# ==========================

IMG_WIDTH = 160
IMG_HEIGHT = 75
CAPTCHA_LEN = 6

# ==========================
# LOAD MODEL + MAPPING
# ==========================

model = tf.keras.models.load_model("captcha_model_v3_5.keras")

with open("char_mapping.json", "r") as f:
    characters = json.load(f)

num_to_char = {i:c for i,c in enumerate(characters)}

# ==========================
# PREPROCESS
# ==========================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh

# ==========================
# SEGMENT (EQUAL SPLIT)
# ==========================

def segment(img):
    char_width = IMG_WIDTH // CAPTCHA_LEN
    chars = []

    for i in range(CAPTCHA_LEN):
        x1 = i * char_width
        x2 = (i + 1) * char_width

        c = img[:, x1:x2]

        # remove empty space
        coords = cv2.findNonZero(255 - c)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            c = c[y:y+h, x:x+w]

        c = cv2.resize(c, (28,28))

        # crop tighter (optional improvement)
        c = cv2.resize(c, (28,28))
        c = c / 255.0

        chars.append(c)

    return chars

# ==========================
# PREDICT
# ==========================

def predict(image_path):

    img = cv2.imread(image_path)
    thresh = preprocess(img)

    chars = segment(thresh)

    result = ""

    for c in chars:
        c = np.reshape(c, (1,28,28,1))

        pred = model.predict(c, verbose=0)
        idx = np.argmax(pred)

        char = num_to_char[idx]   # ✅ FIXED
        result += char

    return result

# ==========================
# TEST
# ==========================

test_image = "captcha_dataset/images/captcha_02011.png"

prediction = predict(test_image)

print("Predicted:", prediction)