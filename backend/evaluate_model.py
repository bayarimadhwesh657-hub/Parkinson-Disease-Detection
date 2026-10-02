import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "model",
    "parkinsons_inceptionv3.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "spiral",
    "testing"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (299, 299)

# IMPORTANT:
# healthy = 0
# parkinson = 1

CLASS_NAMES = {
    0: "Healthy",
    1: "Parkinson's"
}


# ============================================================
# CHECK PATHS
# ============================================================

print("=" * 60)
print("PARKINSON'S MODEL EVALUATION")
print("=" * 60)

print("\nModel path:")
print(MODEL_PATH)

print("\nTesting dataset:")
print(TEST_DIR)


if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    exit()


if not os.path.exists(TEST_DIR):
    print("\nERROR: Testing folder not found!")
    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# DATASET PATHS
# ============================================================

healthy_dir = os.path.join(
    TEST_DIR,
    "healthy"
)

parkinson_dir = os.path.join(
    TEST_DIR,
    "parkinson"
)


if not os.path.exists(healthy_dir):
    print("\nERROR: Healthy folder not found!")
    print(healthy_dir)
    exit()


if not os.path.exists(parkinson_dir):
    print("\nERROR: Parkinson folder not found!")
    print(parkinson_dir)
    exit()


# ============================================================
# GET IMAGE FILES
# ============================================================

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
)


healthy_images = [
    os.path.join(healthy_dir, file)
    for file in os.listdir(healthy_dir)
    if file.lower().endswith(valid_extensions)
]


parkinson_images = [
    os.path.join(parkinson_dir, file)
    for file in os.listdir(parkinson_dir)
    if file.lower().endswith(valid_extensions)
]


print("\nDataset information:")
print("Healthy images:", len(healthy_images))
print("Parkinson images:", len(parkinson_images))
print("Total images:", len(healthy_images) + len(parkinson_images))


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Same preprocessing used by InceptionV3
    img_array = preprocess_input(img_array)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    if probability >= 0.5:
        predicted_class = 1
    else:
        predicted_class = 0

    return predicted_class, probability


# ============================================================
# EVALUATE DATASET
# ============================================================

true_labels = []
predicted_labels = []
probabilities = []


print("\n")
print("=" * 60)
print("EVALUATING HEALTHY IMAGES")
print("=" * 60)


healthy_correct = 0
healthy_wrong = 0


for image_path in healthy_images:

    predicted_class, probability = predict_image(
        image_path
    )

    true_class = 0

    true_labels.append(true_class)
    predicted_labels.append(predicted_class)
    probabilities.append(probability)

    if predicted_class == true_class:

        healthy_correct += 1

    else:

        healthy_wrong += 1

        print(
            "Wrong:",
            os.path.basename(image_path),
            "→",
            CLASS_NAMES[predicted_class],
            f"({probability * 100:.2f}%)"
        )


print("\nHealthy correct:", healthy_correct)
print("Healthy wrong:", healthy_wrong)


print("\n")
print("=" * 60)
print("EVALUATING PARKINSON IMAGES")
print("=" * 60)


parkinson_correct = 0
parkinson_wrong = 0


for image_path in parkinson_images:

    predicted_class, probability = predict_image(
        image_path
    )

    true_class = 1

    true_labels.append(true_class)
    predicted_labels.append(predicted_class)
    probabilities.append(probability)

    if predicted_class == true_class:

        parkinson_correct += 1

    else:

        parkinson_wrong += 1

        print(
            "Wrong:",
            os.path.basename(image_path),
            "→",
            CLASS_NAMES[predicted_class],
            f"({probability * 100:.2f}%)"
        )


print("\nParkinson correct:", parkinson_correct)
print("Parkinson wrong:", parkinson_wrong)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print("\n")
print("Confusion Matrix:")
print(cm)


print("\n")
print("Confusion Matrix Meaning:")

print(
    "Healthy → Healthy       :",
    cm[0][0]
)

print(
    "Healthy → Parkinson's   :",
    cm[0][1]
)

print(
    "Parkinson's → Healthy   :",
    cm[1][0]
)

print(
    "Parkinson's → Parkinson's:",
    cm[1][1]
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Healthy",
            "Parkinson's"
        ],
        zero_division=0
    )
)


print("\nEvaluation completed.")