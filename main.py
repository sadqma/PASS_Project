from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.student_service import get_student_dashboard

from src.predict import predict_single

app = FastAPI(title="PASS - Predictive Academic Success System")

# 👇 вот это добавляем
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # на локальном этапе можно все
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentInput(BaseModel):
    exam_score: Optional[float] = None   # "Балл экз"
    final_score: Optional[float] = None  # "Итог балл"
    rating: float                        # "Рейтинг"


@app.get("/")
def read_root():
    return {
        "message": "PASS API is running",
        "usage": "Send POST to /api/predict with exam_score, final_score, rating"
    }

def api_student_dashboard(student_id: int):
    """
    Возвращает список курсов, прогноз и советы для одного студента.
    Пока без настоящей регистрации/пароля — просто по его StudentId.
    """
    dashboard = get_student_dashboard(student_id)
    return dashboard

@app.post("/api/predict")
def api_predict(student: StudentInput):
    features = {
        "Балл экз": student.exam_score,
        "Итог балл": student.final_score,
        "Рейтинг": student.rating,
    }

    pred = predict_single(features)

    return {
        "input": features,
        "prob_success": pred["prob_success"],
        "prob_success_percent": round(pred["prob_success"] * 100, 1),
        "class": pred["class"],
        "interpretation": "успех" if pred["class"] == 1 else "риск неуспеха",
    }
