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


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred


if __name__ == '__main__':
    X_tra, Y_tra = get_train(PATH)
    X_tst, Y_tst = get_test(PATH)

    Xtr_rows = X_tra.reshape(X_tra.shape[0], 32 * 32 * 3)
    Xte_rows = X_tst.reshape(X_tst.shape[0], 32 * 32 * 3)

    nn = NearestNeighbor()
    nn.train(Xtr_rows, Y_tra)
    Yte_predict = nn.predict(Xte_rows)

    print('accuracy: %f' % (np.mean(Yte_predict == Y_tst)))
