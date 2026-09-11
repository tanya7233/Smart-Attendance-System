import os
import pickle
from typing import List, Tuple

import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def build_embeddings(
    dataset_dir: str,
    prototxt_path: str,
    caffemodel_path: str,
    openface_t7_path: str,
    output_embeddings_path: str,
    detection_confidence: float = 0.5,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Build facial embeddings for all images in dataset_dir and save to pickle.
    dataset_dir should contain subfolders per person with images.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    embedder = cv2.dnn.readNetFromTorch(openface_t7_path)

    image_paths = list(paths.list_images(dataset_dir))
    known_embeddings: List[np.ndarray] = []
    known_names: List[str] = []

    for (i, image_path) in enumerate(image_paths):
        name = image_path.split(os.path.sep)[-2]
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, (min(600, image.shape[1]), int(image.shape[0] * min(600, image.shape[1]) / image.shape[1])))
        (h, w) = image.shape[:2]

        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        detector.setInput(image_blob)
        detections = detector.forward()

        if detections.shape[2] > 0:
            j = int(np.argmax(detections[0, 0, :, 2]))
            confidence = float(detections[0, 0, j, 2])
            if confidence >= detection_confidence:
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()
                known_embeddings.append(vec.flatten())
                known_names.append(name)

    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    with open(output_embeddings_path, "wb") as f:
        pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)

    return known_embeddings, known_names


def train_recognizer(
    embeddings_path: str,
    recognizer_output_path: str,
    label_encoder_output_path: str,
) -> Tuple[SVC, LabelEncoder]:
    """
    Train a linear SVM on precomputed embeddings and save classifier and label encoder pickles.
    """
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["names"]) if data["names"] else []

    if len(set(labels)) < 2:
        raise ValueError("Need at least two classes to train the model.")

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(np.asarray(data["embeddings"]), labels)

    os.makedirs(os.path.dirname(recognizer_output_path), exist_ok=True)
    with open(recognizer_output_path, "wb") as f:
        pickle.dump(recognizer, f)
    with open(label_encoder_output_path, "wb") as f:
        pickle.dump(label_encoder, f)

    return recognizer, label_encoder


