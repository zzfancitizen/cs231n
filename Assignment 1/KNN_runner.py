from KNN.cifair_10_pickle import *
from KNN.k_nearest_neighbor import KNearestNeighbor
import numpy as np

PATH = '../cifar-10-dataset'

Xtr, Ytr = get_train(PATH)
Xte, Yte = get_test(PATH)

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

Xval_rows = Xtr_rows[:1000, :]
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :]
Ytr = Ytr[1000:]

nn = KNearestNeighbor()
nn.train(Xtr_rows, Ytr)

validation_accuracies = []

for k in [1, 3, 5, 10, 20, 50, 100]:
    Yval_predict = nn.predict(Xval_rows, k)
    print('k: %i accuracy: %f' % (k, np.mean(Yval_predict == Yval)))
    validation_accuracies.append((k, np.mean(Yval_predict == Yval)))
