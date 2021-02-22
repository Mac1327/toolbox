from toolbox.roc_auc_curve import roc_auc_curve
import numpy as np


def test_roc_auc_curve():
    y_test, y_pred = np.hstack((np.ones(50) ,np.zeros(50))), np.hstack((np.ones(50) ,np.zeros(50)))
    assert isinstance(roc_auc_curve(y_test, y_pred),  tuple)
    assert (roc_auc_curve(y_test, y_pred)[1] == 1.0)
