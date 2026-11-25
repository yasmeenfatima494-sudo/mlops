import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess(df, scaler="standard", test_size=0.2, seed=42):
    """Return (train_df, test_df). The scaler is fitted on training data only."""
    df = df.drop_duplicates().dropna()
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    if scaler != "none":
        sc = StandardScaler() if scaler == "standard" else MinMaxScaler()
        cols = X_train.columns
        X_train = pd.DataFrame(sc.fit_transform(X_train), columns=cols,
                               index=X_train.index)
        X_test = pd.DataFrame(sc.transform(X_test), columns=cols,
                              index=X_test.index)
    return X_train.assign(target=y_train), X_test.assign(target=y_test)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/dataset.csv")
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--scaler", default="standard",
                   choices=["standard", "minmax", "none"])
    p.add_argument("--test-size", type=float, default=0.2)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    train, test = preprocess(pd.read_csv(a.input), a.scaler, a.test_size)
    train.to_csv(f"{a.outdir}/train.csv", index=False)
    test.to_csv(f"{a.outdir}/test.csv", index=False)
    print(f"Scaler={a.scaler} train={train.shape} test={test.shape}")