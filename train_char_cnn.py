import os, cv2, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

DATASET_PATH = "char_dataset_clean"
IMG_SIZE = 32

images, labels = [], []

for f in os.listdir(DATASET_PATH):
    if not f.endswith(".png"): continue
    label = f.split("_")[0]

    img = cv2.imread(os.path.join(DATASET_PATH, f), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    img = img / 255.0

    images.append(img)
    labels.append(label)

images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# vocab
chars = sorted(list(set(labels)))
char_to_num = {c:i for i,c in enumerate(chars)}
y = np.array([char_to_num[l] for l in labels])

# split
X_train, X_val, y_train, y_val = train_test_split(images, y, test_size=0.1, random_state=42)

# augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# model
model = tf.keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    layers.Conv2D(32, 3, padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(len(chars), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=3)
]

model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=cb
)

model.save("char_cnn_v1.keras")

with open("char_vocab_v1.json", "w") as f:
    json.dump(chars, f)

print("✅ Char CNN optimized")