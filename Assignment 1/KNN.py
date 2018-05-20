import numpy as np
import os

PATH = '../cifar-10-dataset'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def get_train(path):
    i = 1
    X = []
    Y = []

    while i <= 5:
        file = path + os.path.sep + 'data_batch_{0}'.format(i)
        dict = unpickle(file)
        X_tra_temp = dict["data"]
        Y_tra_temp = dict["labels"]
        X.append(X_tra_temp)
        Y.append(Y_tra_temp)
        i += 1

    X_tra = np.array(X).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_tra = np.array(Y).reshape(50000, 1).transpose(0, 1)

    return X_tra, Y_tra


def get_test(path):
    file = path + os.path.sep + 'test_batch'
    dict = unpickle(file)
    X_tst_temp = dict["data"]
    Y_tst_temp = dict["labels"]
    X_tst_temp = np.array(X_tst_temp)
    Y_tst_temp = np.array(Y_tst_temp)
    X_tst = X_tst_temp.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    Y_tst = Y_tst_temp.reshape(10000, 1).transpose(0, 1)

    return X_tst, Y_tst


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        Ypred = np.zeros((num_test, k), dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = []
            for j in range(self.Xtr.shape[0]):
                distances.append([j, np.sum(np.abs(self.Xtr[j, :] - X[i, :]), axis=0)])

            distances = sorted(distances, key=lambda x: x[1])
            distances = np.array(distances).reshape(len(distances), 2)

            Kmin = []
            for j in range(k):
                Kmin.append(distances[j][0])

            Ypred[i] = [self.ytr[x] for x in Kmin]

            print('K is %i, Step %i' % (k, i))
        return Ypred


# test
if __name__ == '__main__':

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
        match = 0
        Yval_predict = nn.predict(Xval_rows, k)
        for i in range(Yval_predict.shape[0]):
            for j in range(k):
                if Yval[i][0] == Yval_predict[i][j]:
                    match += 1
                    break

        acc = match / Yval_predict.shape[0]
        print('K is %s, acc is %f' % (k, acc))
        validation_accuracies.append((k, acc))

    with open('./dataStore/knn_acc.txt', 'w') as file:
        for line in validation_accuracies:
            content = str(line) + '\\r'
            file.write(str(line))
