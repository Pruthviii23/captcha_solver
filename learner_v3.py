import os
import cv2
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ==========================
# SETTINGS
# ==========================

DATASET_PATH = "char_dataset"
IMG_SIZE = 28

# ==========================
# LOAD DATA
# ==========================

images = []
labels = []

for file in os.listdir(DATASET_PATH):

    if not file.endswith(".png"):
        continue

    label = file.split("_")[0]

    img_path = os.path.join(DATASET_PATH, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # ensure consistent size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    img = img / 255.0

    images.append(img)
    labels.append(label)

images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print("Total samples:", len(images))

# ==========================
# CHARACTER MAPPING
# ==========================

characters = sorted(list(set(labels)))
char_to_num = {c:i for i,c in enumerate(characters)}
num_to_char = {i:c for i,c in enumerate(characters)}

num_classes = len(characters)

print("Classes:", num_classes)

# save mapping
with open("char_mapping.json", "w") as f:
    json.dump(characters, f)

# ==========================
# ENCODE LABELS
# ==========================

y = np.array([char_to_num[l] for l in labels])

# ==========================
# TRAIN / TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    images, y, test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ==========================
# MODEL (STABLE BASELINE)
# ==========================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# ==========================
# SAVE
# ==========================

model.save("captcha_model_v3_5.keras")

print("\n✅ Model + mapping saved.")