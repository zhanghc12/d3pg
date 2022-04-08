import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

pred = np.load('confidence.npy')
pred = (pred > 1e-5).astype(int)
labels_test = np.concatenate([np.ones((1000, 1)), 0 * np.ones((1000, 1))], axis=0)
gt = labels_test.astype(int)
precision, recall, f_score, _ = prf(gt, pred, average='binary')
print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
# print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total) * 100))