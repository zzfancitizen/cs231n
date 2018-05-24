import numpy as np


def L_softmax(X, y, W):
    f = W.dot(X)
    f -= np.max(f, axis=0)

    p = np.exp(f) / np.sum(np.exp(f))

    y_idx = list()
    y = np.reshape(y, (-1,))
    idx = np.arange(y.shape[0]).reshape(-1, ).tolist()
    y_idx.append(y)
    y_idx.append(idx)

    loss = np.sum(-np.log(p[y_idx]))

    return loss
