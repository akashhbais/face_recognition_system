import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sqlalchemy import Column, String, JSON, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# PostgreSQL connection URL
DATABASE_URL = "postgresql://postgres:root@localhost/face_db"

Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    
    employee_id = Column(String, primary_key=True, index=True)  # Unique Employee ID
    name = Column(String, nullable=False)
    image_path = Column(String, unique=True, nullable=False)  # Store image path
    face_embedding = Column(JSON, nullable=False)  # Store DeepFace face embedding as JSON list

import uuid  # ✅ Import UUID module

class CheckIn(Base):
    __tablename__ = "checkins"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))  # ✅ Auto-generate UUID
    employee_id = Column(String, ForeignKey("employees.employee_id"), nullable=False)
    checkin_time = Column(DateTime, default=datetime.utcnow)

    # Relationship with Employee table
    employee = relationship("Employee")


# Database connection setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)
