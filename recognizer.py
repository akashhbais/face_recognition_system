import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
import mediapipe as mp  # ✅ Added for face detection
from deepface import DeepFace
from sqlalchemy.orm import Session
from models import SessionLocal, Employee
from sklearn.preprocessing import normalize
from functools import lru_cache

EMPLOYEE_FOLDER = "employee_faces/"

mp_face_detection = mp.solutions.face_detection
mp_face = mp_face_detection.FaceDetection(min_detection_confidence=0.8)

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Image not found or cannot be loaded.")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face_img = img[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160, 160))
            cv2.imwrite(image_path, face_img)
            return image_path

    print("❌ No face detected in image.")
    return None

@lru_cache(maxsize=1)
def load_known_faces():
    db = SessionLocal()
    employees = db.query(Employee).all()
    db.close()

    if not employees:
        return [], []

    known_face_encodings = []
    known_face_names = []

    for emp in employees:
        try:
            embedding = np.array(emp.face_embedding)
            if embedding.shape[0] == 128:
                embedding = normalize([embedding])[0]
                known_face_encodings.append(embedding)
                known_face_names.append(emp.name)
        except Exception as e:
            print(f"Error loading employee {emp.name}: {str(e)}")

    return known_face_encodings, known_face_names

def recognize_face(image_path):
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return {"message": "❌ Face detection failed. No recognizable face found."}

    known_encodings, known_names = load_known_faces()

    if not known_encodings:
        return {"message": "❌ No registered employees found in the system"}

    try:
        test_embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")[0]["embedding"]
        test_embedding = normalize([test_embedding])[0]

        similarities = [np.dot(test_embedding, known) for known in known_encodings]
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]

        THRESHOLD = 0.7

        if best_similarity > THRESHOLD:
            confidence = best_similarity * 100
            return {
                "message": f"Employee Recognized: {known_names[best_match_index]}",
                "accuracy": f"{confidence:.2f}%"
            }
        else:
            return {"message": "❌ No matching employee found", "confidence": f"{best_similarity * 100:.2f}%"}

    except Exception as e:
        return {"message": f"❌ Face detection or embedding extraction failed: {str(e)}"}
