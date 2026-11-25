import os
from sklearn.datasets import load_breast_cancer
os.makedirs("data/raw", exist_ok=True)
df = load_breast_cancer(as_frame=True).frame # 30 features + 'target' column
df.to_csv("data/raw/dataset.csv", index=False)
print("Saved data/raw/dataset.csv with shape", df.shape)