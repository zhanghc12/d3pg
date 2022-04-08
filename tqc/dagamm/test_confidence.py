import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--thre", default=1e-5, type=float)  # Policy name (TD3, DDPG or OurDDPG, Dueling)
args = parser.parse_args()

pred = np.load('confidence.npy')
pred = (pred > args.thre).astype(int)
labels_test = np.concatenate([np.ones((198372, 1)), 0 * np.ones((97278, 1))], axis=0)
gt = labels_test.astype(int)
precision, recall, f_score, _ = prf(gt, pred, average='binary')
print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
# print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total) * 100))