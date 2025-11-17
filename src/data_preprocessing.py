import pandas as pd

def load_data(path="data/DataScienceFFS.csv"):
    # 1️⃣ читаем оригинальный файл в правильной кодировке
    df = pd.read_csv(path, sep=";", encoding="utf-8")

    # 2️⃣ перевод названий колонок
    df = df.rename(columns={
        "YearBegin": "year",
        "Semester": "semester",
        "Course": "course",
        "isStudentType": "student_type",
        "PayName": "payment_type",
        "DisciplineName": "discipline",
        "Валл экз": "exam_score",
        "Итог балл": "final_score",
        "Итоговая": "final_grade",
        "Дата контроля": "control_date",
        "Рейтинг": "rating"
    })

    # 3️⃣ перевод названий дисциплин
    translation_map = {
        "Физическая культура": "Physical Education",
        "Математика": "Mathematics",
        "Инженерная графика в промышленности": "Engineering Graphics in Industry",
        "Устройство автомобилей": "Automotive Design",
        "Информационно-коммуникационные технологии (на англ.)": "Information and Communication Technologies (English)",
        "Физика": "Physics",
        "Химия": "Chemistry",
        "Экономика": "Economics",
        "Основы программирования": "Programming Fundamentals",
        "Инженерная графика": "Engineering Graphics",
        "Теоретическая механика": "Theoretical Mechanics",
        "Материаловедение": "Materials Science",
        "Иностранный язык": "Foreign Language",
        "Русский язык": "Russian Language",
        "Казахский язык": "Kazakh Language",
        "История Казахстана": "History of Kazakhstan",
        "Философия": "Philosophy",
        "Метрология, стандартизация и сертификация": "Metrology, Standardization, and Certification"
    }
    if "discipline" in df.columns:
        df["discipline"] = df["discipline"].replace(translation_map)

    # 4️⃣ Очистка данных: удаляем %, пробелы и конвертируем строки в числа
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # 5️⃣ Приводим ключевые числовые колонки к числовому типу
    for col in ["exam_score", "final_score", "rating", "final_grade"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6️⃣ Генерация фичей для визуализаций и модели
    df["attendance"] = df["exam_score"].apply(lambda x: 100 if x >= 85 else 80 if x >= 70 else 60 if pd.notna(x) else None)
    df["assignments_completed"] = df["final_score"].apply(lambda x: 100 if x >= 85 else 80 if x >= 70 else 60 if pd.notna(x) else None)
    df["participation"] = df["rating"].apply(lambda x: 100 if x >= 85 else 80 if x >= 70 else 60 if pd.notna(x) else None)

    df["LMS_activity"] = df[["attendance", "assignments_completed", "participation"]].mean(axis=1)

    # 7️⃣ Бинарный флаг успеха
    df["success"] = df["final_grade"].apply(lambda x: 1 if pd.notna(x) and x >= 4 else 0)

    # 8️⃣ Сортировка
    df = df.sort_values(by=["year", "discipline", "final_score"], ascending=[True, True, False])

    print("✅ Dataset loaded and processed successfully!")
    print(f"🔹 Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"📊 Columns: {list(df.columns)}")

    return df
