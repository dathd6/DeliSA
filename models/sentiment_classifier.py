from sklearn.metrics import accuracy_score, recall_score, f1_score

class SentimentClassifier():
    def __init__(self, classifier, params) -> None:
        self.classifier = classifier(**params)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        self.X_test = X_test
        self.y_pred = self.classifier.predict(X_test)
        return self.y_pred

    def calculate_metrics(self, y_test):
        self.y_test = y_test
        self.metrics = accuracy_score(y_test, self.y_pred)
        return self.metrics