from flask import Flask, jsonify, request
from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

import numpy as np
import cv2
import os
from io import BytesIO


app = Flask(__name__)
CORS(app)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "parkinsons_inceptionv3.keras"
)

# Your previously selected Parkinson prediction threshold
PREDICTION_THRESHOLD = 0.30

# OpenCV spiral validation settings
MIN_CONTOUR_AREA = 500
MIN_EDGE_DENSITY = 0.005
MAX_EDGE_DENSITY = 0.35


# ============================================================
# LOAD PARKINSON MODEL
# ============================================================

model = None

if os.path.exists(MODEL_PATH):

    print("Loading Parkinson's model...")

    model = load_model(MODEL_PATH)

    print("Parkinson's model loaded successfully.")
    print(
        "Parkinson prediction threshold:",
        PREDICTION_THRESHOLD
    )

else:

    print("ERROR: Parkinson model file not found!")

    print("Expected location:")
    print(MODEL_PATH)


# ============================================================
# SPIRAL VALIDATION USING OPENCV
# ============================================================

def validate_spiral_image(image_bytes):
    """
    Performs a basic spiral-drawing validation using OpenCV.

    This is a heuristic guard, not a trained spiral classifier.

    Returns:
        is_spiral_like : bool
        score          : float
        reason         : str
    """

    try:

        # ----------------------------------------------------
        # Convert uploaded bytes into OpenCV image
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        img = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return (
                False,
                0.0,
                "Unable to read the uploaded image."
            )


        # ----------------------------------------------------
        # Resize while preserving reasonable processing size
        # ----------------------------------------------------

        max_dimension = 800

        height, width = img.shape[:2]

        if max(height, width) > max_dimension:

            scale = max_dimension / max(height, width)

            img = cv2.resize(
                img,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )


        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Check image dimensions
        # ----------------------------------------------------

        height, width = gray.shape

        if height < 100 or width < 100:

            return (
                False,
                0.0,
                "Image is too small."
            )


        # ----------------------------------------------------
        # Detect whether image is mostly a screenshot/photo
        # with large rectangular structures
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_pixels = np.count_nonzero(edges)

        total_pixels = edges.shape[0] * edges.shape[1]

        edge_density = (
            edge_pixels / total_pixels
            if total_pixels > 0
            else 0
        )


        # ----------------------------------------------------
        # Threshold image
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5
        )


        # ----------------------------------------------------
        # Remove tiny noise
        # ----------------------------------------------------

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )


        # ----------------------------------------------------
        # Find contours
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        if not contours:

            return (
                False,
                0.0,
                "No drawing structure detected."
            )


        # ----------------------------------------------------
        # Analyze significant contours
        # ----------------------------------------------------

        significant_contours = []

        image_area = height * width

        for contour in contours:

            area = cv2.contourArea(contour)

            if area >= MIN_CONTOUR_AREA:

                significant_contours.append(
                    contour
                )


        if not significant_contours:

            return (
                False,
                0.0,
                "No significant drawing detected."
            )


        # ----------------------------------------------------
        # Largest contour
        # ----------------------------------------------------

        largest_contour = max(
            significant_contours,
            key=cv2.contourArea
        )

        largest_area = cv2.contourArea(
            largest_contour
        )

        largest_area_ratio = (
            largest_area / image_area
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        bbox_area = w * h

        bbox_ratio = (
            bbox_area / image_area
            if image_area > 0
            else 0
        )


        # ----------------------------------------------------
        # Spiral-like properties
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            largest_contour,
            True
        )

        if perimeter <= 0:

            return (
                False,
                0.0,
                "Invalid contour."
            )


        # Contour complexity
        contour_complexity = (
            perimeter * perimeter
        ) / (
            4 * np.pi * largest_area
            + 1e-8
        )


        # ----------------------------------------------------
        # Calculate circularity
        # ----------------------------------------------------

        circularity = (
            4 * np.pi * largest_area
        ) / (
            perimeter * perimeter
            + 1e-8
        )


        # ----------------------------------------------------
        # Calculate aspect ratio
        # ----------------------------------------------------

        aspect_ratio = (
            w / h
            if h > 0
            else 0
        )


        # ----------------------------------------------------
        # Create heuristic score
        # ----------------------------------------------------

        score = 0.0


        # Drawing should occupy a reasonable portion
        # of the image.
        if largest_area_ratio > 0.01:
            score += 0.20

        if largest_area_ratio > 0.03:
            score += 0.10


        # Spiral drawings usually have substantial contour
        # complexity.
        if contour_complexity > 2:
            score += 0.15

        if contour_complexity > 5:
            score += 0.15


        # Reasonable aspect ratio
        if 0.50 <= aspect_ratio <= 2.00:
            score += 0.10


        # Avoid extremely sparse images
        if edge_density >= MIN_EDGE_DENSITY:
            score += 0.10


        # Avoid extremely dense images such as many-text
        # screenshots.
        if edge_density <= MAX_EDGE_DENSITY:
            score += 0.10


        # A moderate circularity value can be consistent
        # with a curved spiral rather than a rectangle.
        if 0.02 <= circularity <= 0.75:
            score += 0.10


        # ----------------------------------------------------
        # Screenshot / rectangular structure rejection
        # ----------------------------------------------------

        rectangular_score = 0.0

        approx = cv2.approxPolyDP(
            largest_contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            rectangular_score += 0.50


        # If contour almost completely fills the image and
        # is rectangular, it is probably not a spiral.
        if (
            len(approx) == 4
            and bbox_ratio > 0.70
        ):

            rectangular_score += 0.50


        if rectangular_score >= 0.80:

            return (
                False,
                round(score, 4),
                "Image appears to contain a large rectangular structure."
            )


        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        spiral_threshold = 0.50

        is_spiral_like = (
            score >= spiral_threshold
        )


        if is_spiral_like:

            reason = (
                "Image passed the spiral drawing "
                "structure check."
            )

        else:

            reason = (
                "Image does not appear to be "
                "a spiral drawing."
            )


        return (
            is_spiral_like,
            round(score, 4),
            reason
        )


    except Exception as e:

        print(
            "Spiral validation error:",
            str(e)
        )

        return (
            False,
            0.0,
            "Unable to validate the image."
        )


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Parkinson's Detection API is running",
        "model_loaded": model is not None,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "spiral_validation": True
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "success",
        "message": "Backend is connected successfully",
        "model_loaded": model is not None,
        "spiral_validation": True,
        "prediction_threshold": PREDICTION_THRESHOLD
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check Parkinson model
    # --------------------------------------------------------

    if model is None:

        return jsonify({
            "status": "error",
            "message": "Parkinson model is not loaded."
        }), 500


    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "status": "error",
            "message": "No image uploaded."
        }), 400


    uploaded_image = request.files["image"]


    if uploaded_image.filename == "":

        return jsonify({
            "status": "error",
            "message": "No image selected."
        }), 400


    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_bytes = uploaded_image.read()


        if not image_bytes:

            return jsonify({
                "status": "error",
                "message": "Uploaded image is empty."
            }), 400


        # ====================================================
        # STEP 1: SPIRAL VALIDATION
        # ====================================================

        (
            is_spiral,
            spiral_score,
            validation_reason
        ) = validate_spiral_image(
            image_bytes
        )


        print(
            "Spiral validation:",
            is_spiral
        )

        print(
            "Spiral score:",
            spiral_score
        )

        print(
            "Validation reason:",
            validation_reason
        )


        # ----------------------------------------------------
        # Reject non-spiral images
        # ----------------------------------------------------

        if not is_spiral:

            return jsonify({
                "status": "error",
                "type": "invalid_image",
                "message": (
                    "Please upload a valid spiral "
                    "drawing image."
                ),
                "spiral_score": spiral_score,
                "validation_reason": validation_reason
            }), 400


        # ====================================================
        # STEP 2: PARKINSON MODEL
        # ====================================================

        img = image.load_img(
            BytesIO(image_bytes),
            target_size=(299, 299)
        )


        img_array = image.img_to_array(
            img
        )


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # InceptionV3 preprocessing
        img_array = preprocess_input(
            img_array
        )


        prediction = model.predict(
            img_array,
            verbose=0
        )


        probability = float(
            prediction[0][0]
        )


        # ----------------------------------------------------
        # Parkinson classification
        # ----------------------------------------------------

        if probability >= PREDICTION_THRESHOLD:

            result = "Parkinson's"

        else:

            result = "Healthy"


        print(
            "Parkinson probability:",
            probability
        )

        print(
            "Final prediction:",
            result
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "status": "success",

            "prediction": result,

            "probability": probability,

            "probability_percentage": round(
                probability * 100,
                2
            ),

            "threshold": PREDICTION_THRESHOLD,

            "spiral_valid": True,

            "spiral_score": spiral_score

        })


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )


        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )