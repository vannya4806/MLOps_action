import numpy as np
from src.train import train_model

def test_train_model_runs():
    X = np.array([[0,0],[1,1],[1,0],[0,1]])
    y = np.array([0,1,1,0])
    
    model = train_model(X, y)
    assert model is not None

def test_model_accuracy():
    X = np.array([[0,0],[1,1],[1,0],[0,1]])
    y = np.array([0,1,1,0])
    
    model = train_model(X, y)
    acc = model.score(X, y)
    assert acc > 0.8
