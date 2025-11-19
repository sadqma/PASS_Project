from pathlib import Path
import joblib
import pandas as pd

# путь к сохранённой модели
MODEL_PATH = Path("data/models/logreg_model.pkl")

# путь к обработанному датасету (для вычисления средних значений)
DATA_PATH = Path("data/processed/students_features.csv")

# имена колонок с признаками
FEATURE_COLS = ["Балл экз", "Итог балл", "Рейтинг"]

# кешируем модель и средние значения в памяти
_model = None
_feature_means = None


def load_model():
    """Ленивая загрузка модели из файла."""
    global _model
    if _model is None:
        print("Loading model from:", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


def get_feature_means():
    """
    Считаем средние значения по признакам, чтобы заполнять пропуски (NaN).
    """
    global _feature_means
    if _feature_means is None:
        df = pd.read_csv(DATA_PATH)
        _feature_means = df[FEATURE_COLS].mean()
    return _feature_means


def predict_single(features: dict) -> dict:
    """
    Делает предсказание для одного студента.

    features ожидается в формате:
      {
        "Балл экз": 80,
        "Итог балл": 75,
        "Рейтинг": 70
      }

    Возвращает:
      {
        "prob_success": 0.8,  # вероятность успеха (класс 1)
        "class": 1            # 1 = успех, 0 = неуспех
      }
    """
    model = load_model()

    # создаём DataFrame с одной строкой и нужным порядком колонок
    df = pd.DataFrame([features], columns=FEATURE_COLS)

    # заполняем пропуски средними значениями по датасету
    means = get_feature_means()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(means)

    # предсказываем вероятность класса 1
    prob = model.predict_proba(df)[0, 1]
    pred_class = int(model.predict(df)[0])

    return {
        "prob_success": float(prob),
        "class": pred_class,
    }


if __name__ == "__main__":
    # небольшой тест
    example = {
        "Балл экз": 80,
        "Итог балл": 75,
        "Рейтинг": 70,
    }
    print(predict_single(example))
