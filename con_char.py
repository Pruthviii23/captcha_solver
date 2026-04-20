import os
import cv2
import numpy as np

# ==========================
# SETTINGS
# ==========================

DATASET_PATH = "dataset"
OUTPUT_PATH = "char_dataset_clean"

IMG_WIDTH = 160
IMG_HEIGHT = 75
CAPTCHA_LEN = 6

CHAR_SIZE = 28

os.makedirs(OUTPUT_PATH, exist_ok=True)

counter = 0

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
# SEGMENT USING CONTOURS
# ==========================

def segment_contours(thresh):

    # invert (so text becomes white for contour detection)
    thresh_inv = 255 - thresh

    contours, _ = cv2.findContours(
        thresh_inv,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 🔥 filter noise (very important)
        if w < 5 or h < 15:
            continue

        if h > IMG_HEIGHT:  # avoid weird full blobs
            continue

        candidates.append((x, y, w, h))

    # sort left → right
    candidates = sorted(candidates, key=lambda b: b[0])

    # ==========================
    # HANDLE DIFFERENT CASES
    # ==========================

    if len(candidates) < CAPTCHA_LEN:
        return None  # fallback later

    # if too many, pick largest 6 by area
    if len(candidates) > CAPTCHA_LEN:
        candidates = sorted(candidates, key=lambda b: b[2]*b[3], reverse=True)
        candidates = candidates[:CAPTCHA_LEN]
        candidates = sorted(candidates, key=lambda b: b[0])

    # extract characters
    chars = []

    for (x, y, w, h) in candidates:
        char = thresh[y:y+h, x:x+w]

        # pad to make square
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2

        padded[y_offset:y_offset+h, x_offset:x_offset+w] = char

        char = cv2.resize(padded, (CHAR_SIZE, CHAR_SIZE))
        char = char / 255.0

        chars.append(char)

    if len(chars) != CAPTCHA_LEN:
        return None

    return chars


# ==========================
# FALLBACK (SAFE)
# ==========================

def segment_equal(thresh):
    char_width = IMG_WIDTH // CAPTCHA_LEN
    chars = []

    for i in range(CAPTCHA_LEN):
        x1 = i * char_width
        x2 = (i + 1) * char_width

        c = thresh[:, x1:x2]

        c = cv2.resize(c, (CHAR_SIZE, CHAR_SIZE))
        c = c / 255.0

        chars.append(c)

    return chars


# ==========================
# MAIN LOOP
# ==========================

bad_samples = 0

for file in os.listdir(DATASET_PATH):

    if not file.endswith(".png"):
        continue

    label = file.split(".")[0]

    if len(label) != CAPTCHA_LEN:
        continue

    path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(path)
    if img is None:
        continue

    thresh = preprocess(img)

    chars = segment_contours(thresh)

    # fallback if segmentation fails
    if chars is None:
        bad_samples += 1
        chars = segment_equal(thresh)

    # save chars
    for i, char_img in enumerate(chars):
        char_label = label[i]

        filename = f"{char_label}_{counter}.png"
        save_path = os.path.join(OUTPUT_PATH, filename)

        cv2.imwrite(save_path, (char_img * 255).astype(np.uint8))

        counter += 1


print("\n==========================")
print("Done.")
print("Total characters:", counter)
print("Fallback used:", bad_samples)
print("==========================")