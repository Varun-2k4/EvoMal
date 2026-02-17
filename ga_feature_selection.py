import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def genetic_feature_selection(X, y, generations=5, population_size=6):

    num_features = X.shape[1]

    population = np.random.randint(2, size=(population_size, num_features))

    selection_frequency = np.zeros(num_features)

    for _ in range(generations):

        scores = []

        for individual in population:

            if np.sum(individual) == 0:
                scores.append(0)
                continue

            selected = X[:, individual == 1]

            model = RandomForestClassifier(n_estimators=30)
            model.fit(selected, y)

            preds = model.predict(selected)

            score = f1_score(y, preds, average="weighted")
            scores.append(score)

        best_indices = np.argsort(scores)[-2:]
        population = population[best_indices]

        for ind in population:
            selection_frequency += ind

        # Crossover
        child = (population[0] + population[1]) > 0
        population = np.vstack([population, child.astype(int)])

    best_mask = population[np.argmax(scores)]

    return best_mask, selection_frequency
