from typing import List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from PIL import Image



def run_yunet_cv2(
    frame_bgr: np.ndarray,
    model_path: Path,
    *,
    score_thresh: float = 0.3,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[List[Tuple[float, float]]]]:
    """Detects a face and keypoints in an image using OpenCV's YuNet wrapper.

    This function loads a YuNet face detector, runs detection on the given frame, 
    and returns the bounding box and facial keypoints for the most confident detection.

    Args:
        frame_bgr (np.ndarray): Input image in BGR colour format.
        model_path (Path): Path to the YuNet ONNX model.
        score_thresh (float, optional): Minimum confidence threshold for detections. 
            Defaults to 0.3.

    Returns:
        Tuple[Optional[Tuple[float, float, float, float]], Optional[List[Tuple[float, float]]]]:
            - Bounding box as (x, y, width, height) in pixel coordinates.
            - List of 5 keypoints as (x, y) tuples in pixel coordinates.
            - Returns (None, None) if no face is detected or an error occurs.
    """
    try:
        if not model_path.exists():
            return None, None

        height, width = frame_bgr.shape[:2]

        # Create YuNet face detector
        detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (width, height),
            score_threshold=float(score_thresh),
            nms_threshold=0.3,
            top_k=5000,
        )
        detector.setInputSize((width, height))

        # Run detection
        retval, faces = detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None, None

        # Select the face with the highest confidence
        faces = faces.reshape(-1, 15)
        best_idx = int(np.argmax(faces[:, 14]))
        best_face = faces[best_idx]

        # Extract bounding box
        x, y, box_w, box_h = [float(v) for v in best_face[:4]]
        face_box = (x, y, box_w, box_h)

        # Extract 5 keypoints
        keypoints = [
            (float(best_face[4]), float(best_face[5])),   # Right eye
            (float(best_face[6]), float(best_face[7])),   # Left eye
            (float(best_face[8]), float(best_face[9])),   # Nose tip
            (float(best_face[10]), float(best_face[11])), # Right mouth corner
            (float(best_face[12]), float(best_face[13])), # Left mouth corner
        ]

        return face_box, keypoints

    except Exception:
        return None, None

def detect_eyes_in_roi(
    frame_bgr: np.ndarray,
    box: Tuple[float, float, float, float],
) -> Optional[List[Tuple[float, float]]]:
    """Detects two eyes within a given face bounding box using Haar cascade.

    This function serves as a fallback when more advanced detectors fail.
    It converts the region of interest (ROI) to grayscale, applies a Haar
    cascade classifier for eye detection, and returns the pixel coordinates
    of the two largest detected eyes.

    Args:
        frame_bgr (np.ndarray): Input image in BGR colour format.
        box (Tuple[float, float, float, float]): Face bounding box 
            as (x, y, width, height).

    Returns:
        Optional[List[Tuple[float, float]]]:
            - A list of two (x, y) tuples representing the centres of the detected eyes.
            - Returns None if fewer than two eyes are detected or on error.
    """
    # Convert bounding box to integer pixel coordinates
    x, y, w, h = [int(v) for v in box]

    # Extract grayscale face region
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray = gray_frame[y : y + h, x : x + w]
    if roi_gray.size == 0:
        return None

    try:
        # Load Haar cascade for eye detection
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
    except Exception:
        return None

    # Detect eyes inside the ROI
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
    )

    if len(eyes) < 2:
        return None

    # Select the two largest detections by area
    eyes = sorted(eyes, key=lambda r: r[2] * r[3], reverse=True)[:2]

    # Compute centres of detected eyes in full-frame coordinates
    eye_centres: List[Tuple[float, float]] = []
    for (ex, ey, ew, eh) in eyes:
        cx = x + ex + ew / 2.0
        cy = y + ey + eh / 2.0
        eye_centres.append((float(cx), float(cy)))

    # Sort by horizontal position (left eye first, then right eye)
    eye_centres = sorted(eye_centres, key=lambda p: p[0])

    return eye_centres


# ---------- helpers ----------
def draw_debug_overlay(
    frame_bgr: np.ndarray,
    box: Optional[Tuple[float, float, float, float]],
    pts: Optional[List[Tuple[float, float]]],
) -> Image.Image:
    img = frame_bgr.copy()
    if box is not None:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
    if pts:
        for (px, py) in pts:
            cv2.circle(img, (int(px), int(py)), 2, (0, 255, 0), -1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)