import cv2, json, numpy as np, tensorflow as tf

IMG_W, IMG_H = 160, 75

model = tf.keras.models.load_model("captcha_model_hybrid_v2.keras")

with open("char_vocab_hybrid_v2.json") as f:
    chars = json.load(f)

num_to_char = tf.keras.layers.StringLookup(vocabulary=chars, invert=True, mask_token=None)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th / 255.0

def decode(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]

    text = ""
    for i in decoded[0]:
        if i != -1:
            text += num_to_char(i+1).numpy().decode()

    return text

def predict(path):
    img = cv2.imread(path)
    img = preprocess(img)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.reshape(1, IMG_H, IMG_W, 1)

    pred = model.predict(img)
    return decode(pred)

print("Pred:", predict("captcha_dataset/images/captcha_02011.png"))
print("Pred:", predict("captcha_dataset/images/captcha_02012.png"))
print("Pred:", predict("captcha_dataset/images/captcha_02013.png"))
print("Pred:", predict("captcha_dataset/images/captcha_02014.png"))
print("Pred:", predict("captcha_dataset/images/captcha_02015.png"))