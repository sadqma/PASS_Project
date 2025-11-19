from pathlib import Path
from typing import Dict

import pandas as pd

from src.predict import predict_single

DATA_PATH = Path("data/processed/students_features.csv")

# те же имена, что в preprocess_dataset.py
COL_STUDENT = "StudentId"
COL_COURSE = "DisciplineName"
COL_EXAM = "Балл экз"
COL_FINAL = "Итог балл"
COL_RATING = "Рейтинг"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def make_advice(row: pd.Series, prob: float) -> str:
    """Простые текстовые рекомендации по одному курсу."""
    msgs = []

    if row[COL_EXAM] < 70:
        msgs.append("Подтяни экзаменационный балл до 70+.")
    if row[COL_FINAL] < 70:
        msgs.append("Усиль текущую успеваемость по заданиям (итоговый балл < 70).")
    if row[COL_RATING] < 70:
        msgs.append("Увеличь рейтинг: участвуй активнее и сдавай всё вовремя.")

    if prob < 0.6:
        msgs.append("Обратись к преподавателю за помощью и составь план улучшения.")

    if not msgs:
        return "Ты на хорошем уровне — продолжай в том же духе 💪"

    return " ".join(msgs)


def get_student_dashboard(student_id: int) -> Dict:
    """
    Возвращает данные для личного кабинета студента:
    все его курсы + прогноз и советы по каждому.
    """
    df = load_data()

    df_st = df[df[COL_STUDENT] == student_id].copy()

    if df_st.empty:
        return {
            "student_id": student_id,
            "courses": []
        }

    courses = []
    for _, row in df_st.iterrows():
        features = {
            COL_EXAM: row[COL_EXAM],
            COL_FINAL: row[COL_FINAL],
            COL_RATING: row[COL_RATING],
        }
        pred = predict_single(features)
        prob = pred["prob_success"]

        courses.append({
            "discipline": row[COL_COURSE],
            "exam_score": float(row[COL_EXAM]),
            "final_score": float(row[COL_FINAL]),
            "rating": float(row[COL_RATING]),
            "final_calc": float(row["final_calc"]),
            "success": int(row["success"]),
            "prob_success": prob,
            "prob_success_percent": round(prob * 100, 1),
            "advice": make_advice(row, prob),
        })

    return {
        "student_id": student_id,
        "courses": courses
    }


if __name__ == "__main__":
    # маленький тест руками, например StudentId=1
    dash = get_student_dashboard(1)
    print(dash)
