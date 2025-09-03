import numpy as np
from src.model import create_model

def train_model(X, y):
    model = create_model()
    model.train(X, y)
    return model


def train_with_dummy_data():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])

    model = train_model(X, y)
    acc = model.evaluate(X, y)

    print(f"Training selesai. Akurasi pada data dummy: {acc:.2f}")
    return model, acc
