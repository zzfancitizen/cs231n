import numpy as np


def eval_numerical_gradient(f, x):
    """
    一个f在x处的数值梯度法的简单实现
    - f是只有一个参数的函数
    - x是计算梯度的点
    """

    fx = f(x)  # 在原点计算函数值
    grad = np.zeros(x.shape)
    h = 0.00001

    # 对x中所有的索引进行迭代
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # 计算x+h处的函数值
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h  # 增加h
        fxh = f(x)  # 计算f(x + h)
        x[ix] = old_value  # 存到前一个值中 (非常重要)

        # 计算偏导数
        grad[ix] = (fxh - fx) / h  # 坡度
        it.iternext()  # 到下个维度

    return grad
