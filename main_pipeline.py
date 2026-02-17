import numpy as np
from data_preprocessing import load_and_preprocess
from ga_feature_selection import genetic_feature_selection
from hybrid_model import train_lightgbm, train_mlp
from evaluation import evaluate


def run_pipeline(filepath):

    # =============================
    # 1. Load & Preprocess
    # =============================
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(filepath)

    # =============================
    # 2. GA Feature Selection
    # =============================
    mask, selection_freq = genetic_feature_selection(X_train, y_train)

    # Select columns safely using pandas
    selected_columns = X_train.columns[mask == 1]

    X_train_sel = X_train[selected_columns]
    X_test_sel = X_test[selected_columns]

    selected_features = list(selected_columns)

    # =============================
    # 3. Train Hybrid Models
    # =============================
    lgb_model = train_lightgbm(X_train_sel, y_train)
    mlp_model = train_mlp(X_train_sel, y_train)

    # =============================
    # 4. Predictions
    # =============================
    lgb_prob = lgb_model.predict_proba(X_test_sel)[:, 1]
    mlp_prob = mlp_model.predict_proba(X_test_sel)[:, 1]

    final_prob = (lgb_prob + mlp_prob) / 2
    final_preds = (final_prob > 0.5).astype(int)

    # =============================
    # 5. Evaluation
    # =============================
    metrics = evaluate(y_test, final_preds, final_prob)

    # =============================
    # 6. Return
    # =============================
    return (
        metrics,
        lgb_model,
        X_test_sel,
        selected_features,
        selection_freq,
        y_test
    )
