import numpy as np
import os


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

# if __name__ == '__main__':
#
#     Xtr, Ytr = get_train(PATH)
#     Xte, Yte = get_test(PATH)
#
#     Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
#     Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
#
#     Xval_rows = Xtr_rows[:1000, :]
#     Yval = Ytr[:1000]
#     Xtr_rows = Xtr_rows[1000:, :]
#     Ytr = Ytr[1000:]
#
#     nn = KNearestNeighbor()
#     nn.train(Xtr_rows, Ytr)
#
#     validation_accuracies = []
#
#     for k in [1, 3, 5, 10, 20, 50, 100]:
#         Yval_predict = nn.predict(Xval_rows, k)
#         print('k: %i accuracy: %f' % (k, np.mean(Yval_predict == Yval)))
#         validation_accuracies.append((k, np.mean(Yval_predict == Yval)))
