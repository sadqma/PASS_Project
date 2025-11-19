import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/DataScienceFFS.csv")
OUTPUT_PATH = Path("data/processed/unique_disciplines.csv")

COL_COURSE = "DisciplineName"


def main():
    # читаем исходный CSV
    df = pd.read_csv(RAW_PATH, sep=";")
    df.columns = df.columns.str.strip()

    # берем только колонку с предметами, чистим пробелы и NaN
    s = (
        df[COL_COURSE]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # считаем, сколько раз встречается каждый предмет
    uniq = (
        s.value_counts()
         .reset_index()
    )
    uniq.columns = ["DisciplineName", "count"]

    # сортируем по названию (чтобы удобно глазами смотреть)
    uniq = uniq.sort_values("DisciplineName")

    # сохраняем в CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    uniq.to_csv(OUTPUT_PATH, index=False)

    print(f"Unique disciplines: {len(uniq)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
