# roc auc / roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    y_true, y_pred = np.hstack((np.ones(50) ,np.zeros(50))), np.hstack((np.ones(50) ,np.zeros(50)))
    print(roc_auc_score(y_true, y_pred))