import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Путь к нашему обработанному датасету
DATA_PATH = Path("data/processed/students_features.csv")

# Куда будем сохранять модель
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "logreg_model.pkl"

# Какие колонки используем как входные признаки
FEATURE_COLS = ["Балл экз", "Итог балл", "Рейтинг"]
TARGET_COL = "success"


def load_dataset():
    """Читает датасет и возвращает X (features) и y (target)."""
    df = pd.read_csv(DATA_PATH)

    # Печатаем, сколько пропусков в признаках
    print("NaNs before fill:\n", df[FEATURE_COLS].isna().sum())

    # Заполняем пропуски средним значением по колонке
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

    print("NaNs after fill:\n", df[FEATURE_COLS].isna().sum())

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y

def build_model():
    """
    Простая модель:
    StandardScaler + LogisticRegression
    """
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    return pipe


def main():
    print("Loading data from:", DATA_PATH)
    X, y = load_dataset()
    print("Dataset shape:", X.shape)

    # Разбиваем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Оценка качества на тесте
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy on test: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Сохраняем модель
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
