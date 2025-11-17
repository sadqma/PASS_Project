import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Загрузка данных
df = pd.read_csv("D:/498/PASS_Project/data/DataScienceFFS.csv", sep=";")

# 2. Простая очистка
df = df.dropna()

# 3. Создаем целевую переменную success
df["success"] = (df["Итог балл"] >= 70).astype(int)
target = "success"

X = df.drop(columns=[target])
y = df[target]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Определяем типы колонок
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# 6. Препроцессинг
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ==============================
# 7. Логистическая регрессия
# ==============================
log_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500))
])

log_model.fit(X_train, y_train)
pred_log = log_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, pred_log))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred_log))
print("\nClassification report:\n", classification_report(y_test, pred_log))

# Feature importance
log_reg = log_model.named_steps["model"]
importances = log_reg.coef_[0]

feature_names = (
    list(numeric_features) +
    list(log_model.named_steps["preprocess"]
         .named_transformers_["cat"]
         .get_feature_names_out(categorical_features))
)

coef_df = pd.DataFrame({"feature": feature_names, "coef": importances})

print("\nTop 20 feature importance:")
print(coef_df.sort_values("coef", key=np.abs, ascending=False).head(20))

# ==============================
# 8. Модель kNN
# ==============================
knn_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)
pred_knn = knn_model.predict(X_test)

print("\n=== kNN ===")
print("Accuracy:", accuracy_score(y_test, pred_knn))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred_knn))
print("\nClassification report:\n", classification_report(y_test, pred_knn))
