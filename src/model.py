from sklearn.linear_model import LogisticRegression

class MLModel:
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)

def create_model():
    return MLModel()
