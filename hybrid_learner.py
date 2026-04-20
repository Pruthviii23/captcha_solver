import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

DATASET_PATH = "dataset"
IMG_WIDTH = 160
IMG_HEIGHT = 75
MAX_LEN = 6

# ==========================
# LOAD CHAR CNN
# ==========================

char_cnn = tf.keras.models.load_model("char_cnn.keras")

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

    return thresh / 255.0

# ==========================
# LOAD DATA
# ==========================

images, labels = [], []

for file in os.listdir(DATASET_PATH):
    if not file.endswith(".png"):
        continue

    label = file.split(".")[0]

    if len(label) != MAX_LEN:
        continue

    img = cv2.imread(os.path.join(DATASET_PATH, file))
    img = preprocess(img)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    images.append(img)
    labels.append(label)

images = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

print("Loaded:", len(images))

# ==========================
# 🔥 FIXED VOCAB (SAVE ONCE)
# ==========================

if os.path.exists("char_vocab.json"):
    with open("char_vocab.json") as f:
        characters = json.load(f)
else:
    characters = sorted(list(set("".join(labels))))
    with open("char_vocab.json", "w") as f:
        json.dump(characters, f)

num_classes = len(characters) + 1  # CTC blank

char_to_num = {c: i for i, c in enumerate(characters)}
num_to_char = {i: c for i, c in enumerate(characters)}


# ==========================
# 🔥 FIXED ENCODING (NO -1)
# ==========================

def encode(label):
    return [char_to_num[c] for c in label]

encoded = [encode(l) for l in labels]

labels_padded = tf.keras.preprocessing.sequence.pad_sequences(
    encoded,
    maxlen=MAX_LEN,
    padding="post",
    value=0   # IMPORTANT
)

labels_padded = labels_padded.astype(np.int32)

print("Max label:", np.max(labels_padded))
print("Allowed max:", len(characters) - 1)

# ==========================
# 🔥 FIXED CTC LAYER
# ==========================

class CTCLayer(layers.Layer):
    def call(self, y_true, y_pred):

        batch_len = tf.shape(y_true)[0]
        input_len = tf.shape(y_pred)[1]

        # 🔥 REAL label length (ignore padding)
        label_len = tf.reduce_sum(
            tf.cast(tf.not_equal(y_true, 0), tf.int32),
            axis=1,
            keepdims=True
        )

        input_len = tf.cast(input_len, tf.int32)
        batch_len = tf.cast(batch_len, tf.int32)

        input_len = input_len * tf.ones((batch_len, 1), dtype=tf.int32)

        loss = tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_len, label_len
        )

        self.add_loss(loss)
        return y_pred

# ==========================
# MODEL (UNCHANGED CORE)
# ==========================

input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
labels_input = layers.Input(shape=(MAX_LEN,), name="label")

conv1 = layers.Conv2D(32, (3,3), padding="same", activation='relu', name="conv1")(input_img)
pool1 = layers.MaxPooling2D((2,2))(conv1)

conv2 = layers.Conv2D(64, (3,3), padding="same", activation='relu', name="conv2")(pool1)
pool2 = layers.MaxPooling2D((2,2))(conv2)

x = pool2

# ==========================
# SEQUENCE
# ==========================

x = layers.Permute((2,1,3))(x)
x = layers.Reshape((IMG_WIDTH//4, -1))(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.LayerNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

x = layers.Dense(num_classes, activation="softmax")(x)

output = CTCLayer()(labels_input, x)

model = tf.keras.Model([input_img, labels_input], output)
model.compile(optimizer="adam")

model.summary()

for i, layer in enumerate(char_cnn.layers):
    print(i, layer.name, len(layer.get_weights()))

# ==========================
# 🔥 SAFE WEIGHT TRANSFER
# ==========================
# safer: find conv layers dynamically
conv_layers = [l for l in char_cnn.layers if isinstance(l, tf.keras.layers.Conv2D)]

model.get_layer("conv1").set_weights(conv_layers[0].get_weights())
model.get_layer("conv2").set_weights(conv_layers[1].get_weights())

print("✅ Weights transferred")

# ==========================
# TRAIN
# ==========================

model.fit(
    [images, labels_padded],
    validation_split=0.1,   # IMPORTANT
    epochs=40,
    batch_size=32
)

# ==========================
# SAVE
# ==========================

prediction_model = tf.keras.Model(input_img, x)
prediction_model.save("captcha_model.keras")

print("✅ Training complete")