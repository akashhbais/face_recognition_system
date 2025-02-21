import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from sklearn.preprocessing import normalize

from real_time_checkin import capture_face
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import shutil
import numpy as np
import mediapipe as mp  # ✅ Added MediaPipe for better face detection
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from deepface import DeepFace
from models import Employee, CheckIn, SessionLocal
from datetime import datetime
from pydantic import BaseModel
from typing import List
from recognizer import preprocess_image, recognize_face  # ✅ Import updated recognizer function

app = FastAPI()

EMPLOYEE_FOLDER = "employee_faces/"

# Ensure the folder exists
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)

# ✅ Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face = mp_face_detection.FaceDetection(min_detection_confidence=0.8)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class EmployeeCreate(BaseModel):
    employee_id: str
    name: str

class CheckInRequest(BaseModel):
    employee_id: str

@app.post("/employees/")
def create_employee(employee: EmployeeCreate, db: Session = Depends(get_db)):
    existing_employee = db.query(Employee).filter(Employee.employee_id == employee.employee_id).first()
    if existing_employee:
        raise HTTPException(status_code=400, detail="❌ Employee already exists.")

    # Correct image path (directly inside employee_faces/)
    image_filename = f"{employee.employee_id}.jpg"
    image_path = os.path.join(EMPLOYEE_FOLDER, image_filename)  # No subfolder

    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"❌ Employee face image not found at {image_path}. Save the image first.")

    # Preprocess the image (Detect and crop the face)
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        raise HTTPException(status_code=400, detail="❌ No face detected in the image.")

    try:
        # Extract Face Embedding
        embedding = DeepFace.represent(img_path=processed_image, model_name="Facenet")[0]["embedding"]
        embedding = normalize([embedding])[0].tolist()  # Normalize and convert to list

        # Save employee details to the database
        new_employee = Employee(
            employee_id=employee.employee_id,
            name=employee.name,
            image_path=image_path,  # Store correct image path
            face_embedding=embedding
        )

        db.add(new_employee)
        db.commit()
        db.refresh(new_employee)

        return {
            "message": "✅ Employee created successfully!",
            "employee_id": new_employee.employee_id,
            "name": new_employee.name,
            "image_path": new_employee.image_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error extracting face embedding: {str(e)}")


@app.post("/checkin/")
async def checkin_employee(db: Session = Depends(get_db)):
    image_path = capture_face()

    if image_path is None:
        raise HTTPException(status_code=400, detail="❌ Failed to capture face from webcam.")

    recognition_result = recognize_face(image_path)

    if "Employee Recognized" not in recognition_result["message"]:
        raise HTTPException(status_code=401, detail="❌ Face recognition failed. Unauthorized check-in.")

    recognized_name = recognition_result.get("message", "").split(": ")[-1]
    
    # Fetch employee record
    employee = db.query(Employee).filter(Employee.name == recognized_name).first()
    
    if not employee:
        raise HTTPException(status_code=404, detail="❌ Employee record not found in database.")

    # Store check-in record
    checkin_entry = CheckIn(employee_id=employee.employee_id)
    db.add(checkin_entry)
    db.commit()

    os.remove(image_path)  # Cleanup image

    return {
        "message": f"✅ Employee {employee.employee_id} checked in successfully!",
        "recognized_as": recognized_name,
        "accuracy": recognition_result.get("accuracy", "N/A"),
        "time": checkin_entry.checkin_time
    }


@app.get("/checkins/")
def get_checkins(db: Session = Depends(get_db)):
    checkins = db.query(CheckIn).all()
    return checkins
