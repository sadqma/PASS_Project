import pandas as pd

def load_data(path="data/student_data.csv"):
    df = pd.read_csv(path)
    print("\n✅ Data Loaded Successfully!")
    print("\n📊 First 5 Rows:\n", df.head())
    print("\n🔍 Summary Statistics:\n", df.describe())
    return df
