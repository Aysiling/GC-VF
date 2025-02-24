from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from preprocessing import *


def load_graph():
    graphs = get_data()
    return graphs


def find_best_threshold(true_labels, pred_scores):
    thresholds = np.arange(0, 1.01, 0.01)  
    f1_scores = []

    for threshold in thresholds:
        pred_labels = (pred_scores >= threshold).astype(int)
        f1 = f1_score(true_labels, pred_labels)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]

    return best_threshold







