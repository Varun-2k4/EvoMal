import numpy as np
from data_preprocessing import load_and_preprocess
from ga_feature_selection import genetic_feature_selection
from evaluation import evaluate
import lightgbm as lgb


def run_pipeline(filepath):

    # =============================
    # 1. Load Data
    # =============================
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(filepath)

    # =============================
    # 2. GA Feature Selection
    # =============================
    mask, selection_freq = genetic_feature_selection(X_train, y_train)

    X_train_sel = X_train[:, mask == 1]
    X_test_sel = X_test[:, mask == 1]

    selected_features = feature_names[mask == 1]

    # =============================
    # 3. Hybrid Model (LightGBM)
    # =============================
    model = lgb.LGBMClassifier()

    model.fit(X_train_sel, y_train)

    final_preds = model.predict(X_test_sel)
    final_probs = model.predict_proba(X_test_sel)

    # =============================
    # 4. Evaluate
    # =============================
    metrics = evaluate(y_test, final_preds, final_probs)

    return (
        metrics,
        model,
        X_test_sel,
        selected_features,
        selection_freq,
        y_test
    )
