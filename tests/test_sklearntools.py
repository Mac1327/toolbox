from toolbox.roc_auc_curve import roc_auc_curve
from toolbox.trainvaltest import trainvaltest
import numpy as np


def test_roc_auc_curve():
    y_test, y_pred = np.hstack((np.ones(50) ,np.zeros(50))), np.hstack((np.ones(50) ,np.zeros(50)))
    assert isinstance(roc_auc_curve(y_test, y_pred),  tuple)
    assert (roc_auc_curve(y_test, y_pred)[1] == 1.0)

def test_trainvaltest():
    X, y =  np.ones(100), np.ones(100)
    X_train, X_val, X_test, y_train, y_val, y_test = trainvaltest(X, y, test_ratio=0.8)
    assert len(X_val) == len(X_test)
    assert len(X_train) != len(X_test)
    assert (len(X_train)) == 80
    assert (len(X_val)) == 10