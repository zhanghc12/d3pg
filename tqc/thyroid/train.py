import numpy as np
import pandas as pd
import seaborn as sns
from uncertainty_demo import darl2
import torch
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import random

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set(rc={'figure.figsize': [20, 20]}, font_scale=1.4)
try:
    df = pd.read_csv('./tqc/thyroid/hypothyroid.csv')
except:
    df = pd.read_csv('./hypothyroid.csv')
df["binaryClass"]=df["binaryClass"].map({"P":0,"N":1})

df=df.replace({"t":1,"f":0})
del df["TBG"]
df=df.replace({"?":np.NAN})
df=df.replace({"F":1,"M":0})
del df["referral source"]
cols = df.columns[df.dtypes.eq('object')]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df['T4U measured'].fillna(df['T4U measured'].mean(), inplace=True)
df['sex'].fillna(df['sex'].mean(), inplace=True)
df['age'].fillna(df['age'].mean(), inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')

df['TSH'] = imputer.fit_transform(df[['TSH']])
df['T3'] = imputer.fit_transform(df[['T3']])
df['TT4'] = imputer.fit_transform(df[['TT4']])
df['T4U'] = imputer.fit_transform(df[['T4U']])
df['FTI'] = imputer.fit_transform(df[['FTI']])

x = df.drop('binaryClass', axis=1)
y = df['binaryClass']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)
# x_test = sc.transform(x_test)

train_x = np.array(x)
train_y = np.array(y)

normal_data = train_x[train_y == 0]
normal_labels = train_y[train_y == 0]

n_train = int(normal_data.shape[0] * 0.917)
ixs = np.arange(normal_data.shape[0])
np.random.shuffle(ixs)
normal_data_test = normal_data[ixs[n_train:]]
normal_labels_test = normal_labels[ixs[n_train:]]

train_data = normal_data[ixs[:n_train]]
train_label = normal_labels[ixs[:n_train]]
anomalous_data = train_x[train_y == 1]
anomalous_labels = train_y[train_y == 1]

print(len(anomalous_data))
print(len(normal_data_test))
test_data = np.concatenate((anomalous_data, normal_data_test), axis=0)
test_label = np.concatenate((anomalous_labels, normal_labels_test), axis=0)


feature_nn, tree = darl2.build_tree(train_data, dim=40)
print('tree built')
with torch.no_grad():
    confidence = darl2.get_uncertainty(test_data, feature_nn, tree)
print(confidence)

pred = (confidence > 0.099).astype(int)
labels_test = test_label
gt = labels_test.astype(int)
precision, recall, f_score, _ = prf(gt, pred, average='binary')
print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))

# np.save('./tqc/thyroid/confidence.npy',confidence)