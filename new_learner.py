import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

IMG_WIDTH = 160
IMG_HEIGHT = 75
CAPTCHA_LEN = 6

# ==========================
# LOAD DATA
# ==========================

images, labels = [], []

for file in os.listdir("dataset"):
    if not file.endswith(".png"):
        continue

    label = file.split(".")[0]

    if len(label) != CAPTCHA_LEN:
        continue

    img = cv2.imread(os.path.join("dataset", file))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0

    images.append(img)
    labels.append(label)

images = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

# ==========================
# CHAR SET
# ==========================

characters = sorted(list(set("".join(labels))))
num_classes = len(characters)

with open("char_vocab.json", "w") as f:
    json.dump(characters, f)

char_to_num = {c:i for i,c in enumerate(characters)}

# encode labels
y = np.array([[char_to_num[c] for c in label] for label in labels])

# split
X_train, X_val, y_train, y_val = train_test_split(
    images, y, test_size=0.2, random_state=42
)

# ==========================
# MODEL
# ==========================

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

# 6 outputs
outputs = [
    layers.Dense(num_classes, activation="softmax", name=f"char_{i}")(x)
    for i in range(CAPTCHA_LEN)
]

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss=["sparse_categorical_crossentropy"] * CAPTCHA_LEN,
    metrics=["accuracy"] * CAPTCHA_LEN
)

model.summary()

# split labels per position
y_train_split = [y_train[:, i] for i in range(CAPTCHA_LEN)]
y_val_split   = [y_val[:, i] for i in range(CAPTCHA_LEN)]

model.fit(
    X_train,
    y_train_split,
    validation_data=(X_val, y_val_split),
    epochs=25,
    batch_size=32
)

model.save("captcha_model_multiclass.keras")

print("✅ DONE")