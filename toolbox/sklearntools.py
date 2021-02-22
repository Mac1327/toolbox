# roc auc / roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def roc_auc_curve(y_true, y_pred):
    """
    give y_true and y_pred 
    returns roc_auc curve visulalisation 
    and roc score 
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.ylabel('recall')
        plt.plot([0, 1], [0, 1], 'k--') 

    plot_roc_curve(fpr, tpr)
    plt.show()
    return "roc score:", roc_auc_score(y_true, y_pred)


def trainvaltest(X, y, test_ratio=0.8):
    """
    splits X, and y in to X_train, X_val, X_test, y_train, y_val, y_test
    Set ratio with ratio param. 
    """
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, train_size=test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=0.5, test_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test