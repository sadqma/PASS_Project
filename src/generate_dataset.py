# src/generate_dataset.py

import pandas as pd
import numpy as np

np.random.seed(42)

num_students = 50

data = {
    "student_id": range(1, num_students + 1),
    "attendance": np.random.randint(60, 101, num_students),
    "assignments_completed": np.random.randint(50, 101, num_students),
    "participation": np.random.randint(40, 101, num_students),
    "LMS_activity": np.random.randint(100, 1001, num_students),
    "final_grade": np.random.randint(50, 101, num_students)
}

df = pd.DataFrame(data)
df["success"] = (df["final_grade"] >= 70).astype(int)  # 1=success, 0=failure
df.to_csv("data/student_data.csv", index=False)

print("✅ Simulated dataset saved to data/student_data.csv")
