# src/generate_dataset.py

import pandas as pd
import numpy as np

np.random.seed(42)

num_students = 200

# --- Генерация базовых показателей ---
attendance = np.random.normal(85, 10, num_students).clip(50, 100)
assignments_completed = np.random.normal(80, 12, num_students).clip(40, 100)
participation = np.random.normal(75, 15, num_students).clip(30, 100)
study_hours = np.random.normal(10, 3, num_students).clip(1, 20)
sleep_hours = np.random.normal(7, 1.5, num_students).clip(4, 10)
stress_level = np.random.normal(50, 20, num_students).clip(0, 100)
social_media_usage = np.random.normal(3, 1.5, num_students).clip(0.5, 8)
LMS_activity = np.random.randint(100, 1200, num_students)

# --- Формирование итоговой оценки ---
# Весовые коэффициенты (чем больше — тем сильнее влияет на успех)
final_grade = (
    0.25 * attendance +
    0.25 * assignments_completed +
    0.2 * participation +
    0.15 * (study_hours * 10) +
    0.05 * np.log(LMS_activity) * 10 +
    np.random.normal(0, 5, num_students)
)

# Коррекция на стресс и недостаток сна
final_grade -= (stress_level / 50) * 5
final_grade += (sleep_hours - 7) * 3

final_grade = final_grade.clip(0, 100)

# --- Метка успеха ---
success = (final_grade >= 70).astype(int)

# --- Итоговый DataFrame ---
df = pd.DataFrame({
    "student_id": range(1, num_students + 1),
    "attendance": attendance.round(1),
    "assignments_completed": assignments_completed.round(1),
    "participation": participation.round(1),
    "study_hours": study_hours.round(1),
    "sleep_hours": sleep_hours.round(1),
    "stress_level": stress_level.round(1),
    "social_media_usage": social_media_usage.round(1),
    "LMS_activity": LMS_activity,
    "final_grade": final_grade.round(1),
    "success": success
})

df.to_csv("data/student_data.csv", index=False)

print("✅ Realistic student dataset saved to data/student_data.csv")
