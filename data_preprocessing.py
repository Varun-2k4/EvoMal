import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess(filepath):

    # Load dataset
    df = pd.read_csv(filepath)

    # Replace missing values like '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Convert all columns to numeric if possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Automatically detect target column
    possible_targets = ["Label", "label", "class", "Class", "target", "Target"]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]  # last column as default

    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode target if categorical
    if y.dtype == 'object':
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # Keep feature names
    feature_names = X.columns

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame (IMPORTANT FIX)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_names
