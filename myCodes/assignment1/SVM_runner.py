from utils.cifair_10_pickle import *
from linear_classification.multiClassSVM import *
import os
import numpy as np

PATH = os.path.abspath('../cifar-10-dataset')

X_tra, y_tra = get_train(PATH)

X_tra = np.reshape(X_tra, (-1, 32 * 32 * 3)).transpose((1, 0))
y_tra = np.reshape(y_tra, (-1,))
X_tra = X_tra / 255  # regularization

W = np.random.normal(0, .1, (10, 32 * 32 * 3))

loss1 = 0.0

for i in range(X_tra.shape[1]):
    x = X_tra[:, i]
    y = y_tra[i]
    loss1 += L_i(x, y, W)

loss2 = 0.0

for i in range(X_tra.shape[1]):
    x = X_tra[:, i]
    y = y_tra[i]
    loss2 += L_i_vectorized(x, y, W)

loss3 = 0.0

loss3 = L(X_tra, y_tra, W)

print(loss1)
print(loss2)
print(loss3)
