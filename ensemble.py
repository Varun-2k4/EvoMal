import numpy as np

def soft_voting(lgb_model, mlp_model, X):

    p1 = lgb_model.predict_proba(X)[:,1]
    p2 = mlp_model.predict_proba(X)[:,1]

    final_prob = 0.6 * p1 + 0.4 * p2

    return (final_prob > 0.5).astype(int)
