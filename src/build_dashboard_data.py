import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.predict import predict_single

DATA_PATH = Path("data/processed/students_features.csv")
OUTPUT_PATH = Path("web/dashboard_data.json")

COL_STUDENT = "StudentId"
COL_COURSE = "DisciplineName"
COL_EXAM = "Балл экз"
COL_FINAL = "Итог балл"
COL_RATING = "Рейтинг"


def make_advice(row: pd.Series, prob: float) -> str:
    msgs = []

    # если значения NaN, сравнение даст False, так что тут всё ок
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


def safe_num(x):
    """Преобразуем число в float или None (если NaN), чтобы в JSON был null."""
    return float(x) if pd.notna(x) else None


def build_dashboard_data() -> Dict[str, Any]:
    df = pd.read_csv(DATA_PATH)

    data: Dict[str, Any] = {"students": {}}

    for _, row in df.iterrows():
        sid = int(row[COL_STUDENT])
        sid_str = str(sid)

        # исходные значения (могут быть NaN)
        exam = row[COL_EXAM]
        final = row[COL_FINAL]
        rating = row[COL_RATING]
        final_calc = row["final_calc"]

        # для модели оставляем как есть — predict_single сам заполнит NaN средними
        features = {
            COL_EXAM: exam,
            COL_FINAL: final,
            COL_RATING: rating,
        }

        pred = predict_single(features)
        prob = pred["prob_success"]

        student_entry = data["students"].setdefault(
            sid_str, {"student_id": sid, "courses": []}
        )

        student_entry["courses"].append(
            {
                "discipline": row[COL_COURSE],
                "exam_score": safe_num(exam),
                "final_score": safe_num(final),
                "rating": safe_num(rating),
                "final_calc": safe_num(final_calc),
                "success": int(row["success"]),
                "prob_success": prob,
                "prob_success_percent": round(prob * 100, 1),
                "advice": make_advice(row, prob),
            }
        )

    return data


def main():
    data = build_dashboard_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved dashboard data to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
