def load_and_preprocess(filepath):

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # ---------------------------
    # Robust CSV Reading
    # ---------------------------
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except:
        df = pd.read_csv(filepath, encoding="latin1")

    # ---------------------------
    # Basic Cleaning
    # ---------------------------
    df = df.dropna()

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = X.columns

    # Convert to numeric safely
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names
