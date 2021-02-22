from toolbox.trainvaltest import trainvaltest
import numpy as np


def test_trainvaltest():
    X, y =  np.ones(100), np.ones(100)
    X_train, X_val, X_test, y_train, y_val, y_test = trainvaltest(X, y, test_ratio=0.8)
    assert len(X_val) == len(X_test)
    assert len(X_train) != len(X_test)
    assert (len(X_train)) == 80
    assert (len(X_val)) == 10
