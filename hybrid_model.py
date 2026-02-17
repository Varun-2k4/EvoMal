import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

def train_lightgbm(X, y):
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    return model

def train_mlp(X, y):
    model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=50)
    model.fit(X, y)
    return model
