import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def genetic_feature_selection(X, y, population_size=6, generations=4):

    X = np.array(X)
    y = np.array(y)

    n_features = X.shape[1]

    population = np.random.randint(0, 2, (population_size, n_features))
    selection_frequency = np.zeros(n_features)

    for _ in range(generations):

        scores = []

        for individual in population:

            if np.sum(individual) == 0:
                scores.append(0)
                continue

            selected = X[:, individual == 1]

            model = RandomForestClassifier(n_estimators=20)
            model.fit(selected, y)

            preds = model.predict(selected)
            scores.append(f1_score(y, preds, average="weighted"))

        best_indices = np.argsort(scores)[-2:]
        population = population[best_indices]

        selection_frequency += population.sum(axis=0)

    best_mask = population[0]

    return best_mask, selection_frequency
