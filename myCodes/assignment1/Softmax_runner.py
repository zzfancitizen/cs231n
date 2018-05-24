from utils.cifair_10_pickle import *
from linear_classification.Softmax import *
import os
import numpy as np

PATH = os.path.abspath('../cifar-10-dataset')

X_tra, y_tra = get_train(PATH)

X_tra = np.reshape(X_tra, (-1, 32 * 32 * 3)).transpose((1, 0))
y_tra = np.reshape(y_tra, (-1,))
X_tra = X_tra / 255  # regularization

W = np.random.normal(0, .1, (10, 32 * 32 * 3))

loss = L_softmax(X_tra, y_tra, W)

print(loss)
