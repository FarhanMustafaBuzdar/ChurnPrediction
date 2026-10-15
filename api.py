
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import pandas as pd
import pickle

# IMPORT LIBRARIES
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func


# DATABASE CONNECTION


load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

database_url = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(database_url, pool_pre_ping=True)


# CREATE SESSION

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

# CREATE BASE OBJECT

Base = declarative_base()


# DATABASE MODEL

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    Gender = Column(String(100))
    SeniorCitizen = Column(String(100))
    Partner = Column(String(100))
    Dependents = Column(String(100))
    TenureMonths = Column(Integer)
    InternetService = Column(String(100))
    OnlineSecurity = Column(String(100))
    TechSupport = Column(String(100))
    Contract = Column(String(100))
    PaperlessBilling = Column(String(100))
    PaymentMethod = Column(String(100))
    MonthlyCharges = Column(Float)
    TotalCharges = Column(Float)
    Age = Column(Integer)

    prediction = Column(String(100))

    created_at = Column(
        DateTime,
        server_default=func.now()
    )


# CREATE TABLES

Base.metadata.create_all(bind=engine)


# DATABASE SESSION

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# LOAD TRAINED MODEL

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)


# INPUT DATA VALIDATION

class UserInput(BaseModel):
    Gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    TenureMonths: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    Age: int


# FASTAPI APPLICATION

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


# PREDICT AND SAVE

@app.post("/predict")
def predict(
    data: UserInput,
    db: Session = Depends(get_db)
):
    try:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Convert prediction into readable text
        if str(prediction).strip().lower() in ("0", "yes", "true", "churn"):
            result = "Churn"
        else:
            result = "Not Churn"

        # Create database record
        new_prediction = Prediction(
            Gender=data.Gender,
            SeniorCitizen=data.SeniorCitizen,
            Partner=data.Partner,
            Dependents=data.Dependents,
            TenureMonths=data.TenureMonths,
            InternetService=data.InternetService,
            OnlineSecurity=data.OnlineSecurity,
            TechSupport=data.TechSupport,
            Contract=data.Contract,
            PaperlessBilling=data.PaperlessBilling,
            PaymentMethod=data.PaymentMethod,
            MonthlyCharges=data.MonthlyCharges,
            TotalCharges=data.TotalCharges,
            Age=data.Age,
            prediction=result
        )

        # Save record to MySQL
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        return {
            "id": new_prediction.id,
            "prediction": result,
            "created_at": new_prediction.created_at,
            "message": "Prediction saved successfully"
        }

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Prediction or database operation failed."
        )


# VIEW SAVED PREDICTIONS

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    records = (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .all()
    )

    return [
        {
            "id": record.id,
            "prediction": record.prediction,
            "created_at": record.created_at
        }
        for record in records
    ]
