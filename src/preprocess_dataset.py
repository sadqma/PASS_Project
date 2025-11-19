import pandas as pd
from pathlib import Path

# Пути
RAW_PATH = Path("data/raw/DataScienceFFS.csv")
PROCESSED_PATH = Path("data/processed/students_features.csv")
DISC_DIR_PATH = Path("data/processed/discipline_directions.csv")

# Колонки
COL_EXAM = "Балл экз"
COL_FINAL = "Итог балл"
COL_RATING = "Рейтинг"
COL_COURSE = "DisciplineName"
COL_DIRECTION = "Direction"
COL_STUDENT_ID = "StudentId"


def load_raw() -> pd.DataFrame:
    """Читаем сырой датасет."""
    df = pd.read_csv(RAW_PATH, sep=";")
    df.columns = df.columns.str.strip()
    return df


def load_directions_map() -> pd.DataFrame:
    """Читаем маппинг предмет -> направление из discipline_directions.csv."""
    df_map = pd.read_csv(DISC_DIR_PATH)
    df_map.columns = df_map.columns.str.strip()
    df_map = df_map[[COL_COURSE, COL_DIRECTION]].drop_duplicates()
    return df_map


def merge_directions(df: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    """Подмешиваем направление к каждой строке по DisciplineName."""
    df = df.merge(df_map, on=COL_COURSE, how="left")
    df[COL_DIRECTION] = df[COL_DIRECTION].fillna("Другое")
    return df


def fill_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняем пропуски медианой по колонкам с баллами."""
    for col in [COL_EXAM, COL_FINAL, COL_RATING]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def calculate_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Итоговая оценка по формуле:
    40% экзамен, 40% итоговый балл, 20% рейтинг.
    """
    df["final_calc"] = (
        df[COL_EXAM] * 0.40 +
        df[COL_FINAL] * 0.40 +
        df[COL_RATING] * 0.20
    )
    df["success"] = (df["final_calc"] >= 70).astype(int)
    return df


def assign_students(df: pd.DataFrame, courses_per_student: int = 4) -> pd.DataFrame:
    """
    Формируем студентов так, чтобы:
    - у каждого студента НЕ повторялись дисциплины
    - распределение по направлениям (Direction)
    - используем все строки исходного датасета

    Логика:
    - внутри каждого направления перемешиваем строки
    - идём по строкам и кладём каждую в "первого студента",
      у которого ещё есть место (< courses_per_student)
      и у которого ещё нет этого курса
    - если ни к кому не подошло — создаём нового студента
    """
    parts = []
    next_student_id = 1

    for direction, group in df.groupby(COL_DIRECTION):
        # случайно перемешиваем строки, чтобы студенты были "миксом"
        group = group.sample(frac=1, random_state=42).reset_index(drop=True)

        # список множеств предметов для каждого студента внутри направления
        student_courses = []  # List[set[str]]
        row_student_idx = [-1] * len(group)

        for i, row in group.iterrows():
            course = row[COL_COURSE]
            assigned = False

            # пробуем положить строку к уже существующим студентам
            for j, course_set in enumerate(student_courses):
                if len(course_set) < courses_per_student and course not in course_set:
                    row_student_idx[i] = j
                    course_set.add(course)
                    assigned = True
                    break

            # если не нашли подходящего студента — создаём нового
            if not assigned:
                student_courses.append({course})
                row_student_idx[i] = len(student_courses) - 1

        # превращаем локальные индексы студентов в глобальные StudentId
        group[COL_STUDENT_ID] = [next_student_id + idx for idx in row_student_idx]
        next_student_id += len(student_courses)

        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем только нужное для модели и дашборда."""
    return df[
        [
            COL_STUDENT_ID,
            COL_COURSE,
            COL_EXAM,
            COL_FINAL,
            COL_RATING,
            "final_calc",
            "success",
        ]
    ]


def main():
    # 1) сырые данные + направления
    df_raw = load_raw()
    print("Loaded raw:", df_raw.shape)

    df_map = load_directions_map()
    print("Directions map:", df_map.shape)

    # 2) подмешиваем direction и чистим пропуски
    df = merge_directions(df_raw, df_map)
    df = fill_nans(df)

    # 3) считаем итог и success
    df = calculate_final(df)

    # 4) раскладываем строки по студентам (без повторов курсов внутри студента)
    df = assign_students(df, courses_per_student=4)

    # 5) выбираем нужные колонки и сохраняем
    df = select_columns(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Saved to:", PROCESSED_PATH)
    print("Rows:", len(df))
    print(df.head(10))


if __name__ == "__main__":
    main()
