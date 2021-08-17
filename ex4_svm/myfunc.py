import numpy as np


def sigmoid(data):
    if data >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-data))
    else:
        return np.exp(data) / (1 + np.exp(data))


# numpy ufunc函数，输入数组，对每个元素分别操作，输出同维数组
sigmoid = np.frompyfunc(sigmoid, 1, 1)


def expandX(x):
    x_ = np.ndarray(shape=(x.shape[0] + 1, x.shape[1]))
    x_[0, :] = 1
    x_[1:, :] = x
    return x_

def gaussian(x1,x2,sigma2):

    return np.exp(-np.sum(np.square(x1-x2))/2/sigma2)