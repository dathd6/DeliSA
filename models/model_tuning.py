from models.sentiment_classifier import SentimentClassifier
from sklearn.model_selection import GridSearchCV

class ModelTuning(SentimentClassifier):
    def __init__(self, classifier, params, cv=5, scoring='accuracy') -> None:
        # Initialize first parameter
        super().__init__(classifier, params)
        self.grid_search = GridSearchCV(
            classifier(),
            param_grid=params,
            cv=cv,
            scoring=scoring
        )

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.grid_search.fit(X_train, y_train)
        self.classifier = self.grid_search.best_estimator_

    def calculate_metrics(self):
        self.metrics = self.grid_search.best_score_
        return self.metrics