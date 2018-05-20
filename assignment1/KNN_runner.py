from KNN.cifair_10_pickle import *
from KNN.k_nearest_neighbor import KNearestNeighbor
import numpy as np

PATH = '../cifar-10-dataset'

Xtr, Ytr = get_train(PATH)
Xte, Yte = get_test(PATH)

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

nn = KNearestNeighbor()
nn.train(Xtr_rows, Ytr)

validation_accuracies = []

for k in [1, 3, 5, 10, 20, 50, 100]:
    Yte_predict = nn.predict(Xte_rows, k, num_loops=2)
    print('k: %i accuracy: %f' % (k, np.mean(Yte_predict == Yte)))
    validation_accuracies.append((k, np.mean(Yte_predict == Yte)))

with open('./dataStore/knn_acc.txt', 'w') as file:
    for line in validation_accuracies:
        line = str(line) + '\r'
        file.write(line)
