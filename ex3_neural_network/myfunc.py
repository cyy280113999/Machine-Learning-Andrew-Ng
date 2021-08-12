import numpy as np


def sigmoid(data):
    if data >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-data))
    else:
        return np.exp(data) / (1 + np.exp(data))


# numpy ufunc函数，输入数组，对每个元素分别操作，输出同维数组
sigmoid = np.frompyfunc(sigmoid, 1, 1)
fone = np.frompyfunc(lambda x: x, 1, 1)
dfone = np.frompyfunc(lambda x: 1, 1, 1)


def d_sig(data):
    temp = sigmoid(data)
    return temp * (1 - temp)


d_sig = np.frompyfunc(d_sig, 1, 1)

derivative_dictionary = {sigmoid: d_sig,
                         fone: dfone}


def d(fun):
    try:
        return derivative_dictionary[fun]
    except Exception:
        raise


def expandX(x):
    x_ = np.ndarray(shape=(x.shape[0] + 1, x.shape[1]))
    x_[0, :] = 1
    x_[1:, :] = x
    return x_


# 解决log(0)为无穷的问题
error_max = 1e5
mylog = lambda x: np.log(x) if x != 0 else error_max
mylog = np.frompyfunc(mylog, 1, 1)

