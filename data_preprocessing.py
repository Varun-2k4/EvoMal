import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(filepath):

    # -----------------------------
    # Robust CSV Loading
    # -----------------------------
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except:
        df = pd.read_csv(filepath, encoding="latin1")

    df = df.dropna()

    # Assume last column is label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = X.columns.to_numpy()

    # Convert all to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
