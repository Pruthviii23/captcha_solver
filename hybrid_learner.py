import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ==========================
# SETTINGS
# ==========================

DATASET_PATH = "dataset"
IMG_WIDTH = 160
IMG_HEIGHT = 75
MAX_LEN = 6
BATCH_SIZE = 32
EPOCHS = 80

# ==========================
# LOAD DATA
# ==========================

images = []
labels = []

for file in os.listdir(DATASET_PATH):

    if not file.endswith(".png"):
        continue

    label = file.split(".")[0]

    if len(label) != MAX_LEN:
        continue

    img = cv2.imread(os.path.join(DATASET_PATH, file))

    if img is None:
        continue

    # OTSU
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    img = cv2.resize(thresh, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0

    images.append(img)
    labels.append(label)

images = np.array(images, dtype=np.float32)
images = np.expand_dims(images, axis=-1)

print("Dataset:", images.shape)

# ==========================
# CHARSET
# ==========================

characters = sorted(list(set("".join(labels))))

with open("char_vocab.json", "w") as f:
    json.dump(characters, f)

char_to_num = {c:i for i,c in enumerate(characters)}

def encode(label):
    return [char_to_num[c] for c in label]

encoded = np.array([encode(l) for l in labels])

labels_padded = tf.keras.preprocessing.sequence.pad_sequences(
    encoded,
    maxlen=MAX_LEN,
    padding="post",
    value=0
).astype(np.int32)

# ==========================
# SPLIT
# ==========================

X_train, X_val, y_train, y_val = train_test_split(
    images, labels_padded,
    test_size=0.2,
    random_state=42
)

# ==========================
# 🔥 TF.DATA PIPELINE (CRITICAL FIX)
# ==========================

def create_dataset(X, y):

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.map(
        lambda x, y: ({"image": x, "label": y}),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = create_dataset(X_train, y_train)
val_ds   = create_dataset(X_val, y_val)

# ==========================
# CTC LAYER
# ==========================

class CTCLayer(layers.Layer):
    def call(self, y_true, y_pred):

        batch_len = tf.shape(y_true)[0]
        input_len = tf.shape(y_pred)[1]
        label_len = tf.shape(y_true)[1]

        input_len = input_len * tf.ones((batch_len,1), dtype=tf.int32)
        label_len = label_len * tf.ones((batch_len,1), dtype=tf.int32)

        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len
        )

        self.add_loss(loss)
        return y_pred

# ==========================
# MODEL
# ==========================

input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
labels_input = layers.Input(shape=(MAX_LEN,), name="label")

# CNN
x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D(2,2)(x)

x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2,2)(x)

# reshape
x = layers.Permute((2,1,3))(x)
x = layers.Reshape((IMG_WIDTH//2, -1))(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.LayerNormalization()(x)   # 🔥 ADD THIS
x = layers.Dropout(0.3)(x)

# RNN
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

# output
num_classes = len(characters) + 1
x = layers.Dense(num_classes, activation="softmax")(x)

output = CTCLayer()(labels_input, x)

model = tf.keras.Model(
    inputs={"image": input_img, "label": labels_input},
    outputs=output
)

model.compile(optimizer="adam")

model.summary()

# ==========================
# TRAIN
# ==========================

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================
# SAVE
# ==========================

prediction_model = tf.keras.Model(input_img, x)
prediction_model.save("captcha_model.keras")

print("✅ TRAINING COMPLETE")