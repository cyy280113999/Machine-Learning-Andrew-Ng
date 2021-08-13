"""
neural network

模型：
    神经层Layer ，免神经元设计
    神经网Net ，免神经层设计
    正则化损失函数

Net方法：
    getResult
    fitAuto 自动优化学习，使用minimize 和 fmin
    E,E2 铺平输入，常规输入，的误差函数
    vectorizeWeight，铺平参数
    reformatWeight，重构参数
    fit BP学习，含正则化
    fit_batch 多输入学习一次
    gradientCheck 梯度检测
    evaluate 评价成绩
测试：
    createLayer
    createNet
    vectorize 参数向量化测试，自动优化算法需要参数铺平
    netFitAuto 自动优化测试
    netfit 手动优化测试

update：
免神经元设计Layer
免神经层设计，旧Net命名为old_Net，新建Net
平铺参数，测试vectorize，学习fitAuto
手动拟合netfit
梯度检测gradientCheck
fit添加batch_size参数，对于大量数据，分批次学习
"""
from myfunc import *
import numpy as np
import scipy.optimize as op
import numpy.linalg as la
import matplotlib.pyplot as plt


class Layer:
    """
    神经层，元素为神经元
    X是输入，单次输入X为n*1的向量，n是输入大小，
    多次输入X为 n*m 的矩阵，m是多输入数据总量
    Y是输出，单次输入X为k*1的向量
    多次输出Y为 k*m 的矩阵

    [       ]       [       ]
    [   X   ]   --> [   Y   ]
    [       ]n*m    [       ]k*m

    通过矩阵表示输出：
    X扩张为X_，n+1*m的矩阵 第一行全为1

        [1111111]
    X_= [-------]
        [   X   ]
        [       ]n+1*m

    免神经元设计
    ****这里与myNN设计不同，采用WT替代W尝试：
    W是参数，k*n+1的矩阵，每列为各神经元参数，第一列是偏移

        [weight0|w1|..|]
    W = [weight0|  |  |]
        [weight0|  |  |] k*n+1

    [       ]   [       ] [       ]
    [   Y   ] = [   W   ]*[   X_  ]
    [       ]   [       ] [       ]

    """

    def __init__(self, num_input: int, num_output: int, fun=sigmoid):
        """
        :param num_input: n,输入列长
        :param num_output: m,输出列长
        :param fun: 激活函数
        """
        self.num_input = num_input
        self.num_output = num_output
        self.fun = fun
        # self.w = np.ndarray((self.num_output, self.num_input + 1), dtype=np.float64)
        self.w = np.random.random((self.num_output, self.num_input + 1))

    def getResult(self, data_input, w=None):
        if w is None:
            w = self.w
        x = expandX(data_input)
        return self.fun(w @ x)

    def updateW(self, w):
        self.w = w


def createLayer():
    num_in, num_out = 3, 2
    layer = Layer(num_in, num_out)
    data_in = np.random.random((num_in, 1))
    print(layer.getResult(data_in))


class old_Net:
    """
    神经网络，包含层
    输出 Y = Ld(Ld-1(...(L1(X)...)),  eg :L1(X) = sigmod(W*X)
    要生成网络，请给出X的长度，和包含隐含层的神经元个数的列表，列表的长度决定网络的层数
    例：3,[5,6,7]表示三层神经网络，输入X 3*1，输出Y 7*1

    X是输入，单次输入X为n*1的向量，n是输入大小，
    多次输入X为 n*m 的矩阵，m是多输入数据总量
    Y是输出，单次输入X为k*1的向量
    多次输出Y为 k*m 的矩阵

    [       ]       [       ]
    [   X   ]   --> [   Y   ]
    [       ]n*m    [       ]k*m

    """

    def __init__(self, num_input, outputLine):
        self.deep = len(outputLine)
        # self.fun = fun
        self.layerList = np.ndarray((self.deep,), dtype=Layer)
        for index in range(self.deep):
            if index == 0:
                self.layerList[index] = Layer(num_input, outputLine[index])
            elif index == self.deep - 1:
                self.layerList[index] = Layer(outputLine[index - 1], outputLine[index])
            else:
                self.layerList[index] = Layer(outputLine[index - 1], outputLine[index])

    def getResult(self, data_input):
        result = data_input
        for layer in self.layerList:
            result = layer.getResult(result)
        return result

    def show(self):
        """
        这可以展示网络的结构简图
        """
        rows = [''] * 5
        width = 12
        for index, layer in enumerate(self.layerList):
            rows[0] += '|' + "-" * (width - 2) + '|'
            row = "layer {}:".format(index + 1)
            rows[1] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            row = "input:{}".format(layer.num_input)
            rows[2] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            row = "output:{}".format(layer.num_output)
            rows[3] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            rows[4] += '|' + "-" * (width - 2) + '|'
        print('|net:' + ' ' * (width * len(self.layerList) - 6) + '|')
        for row in rows:
            print(row)


def create_old_Net():
    num_in, num_out = 3, 2
    out_list = [4, 5, num_out]
    net = old_Net(num_in, out_list)
    data_in = np.random.random((3, 1))
    print(net.getResult(data_in))
    net.show()


class Net:
    """
    神经网络，包含层
    输出 Y = Ld(Ld-1(...(L1(X)...)),  eg :L1(X) = sigmod(W*X)
    要生成网络，请给出X的长度，和包含隐含层的神经元个数的列表，列表的长度决定网络的层数
    例：3,[5,6,7]表示三层神经网络，输入X 3*1，输出Y 7*1

    X是输入，单次输入X为n*1的向量，n是输入大小，
    多次输入X为 n*m 的矩阵，m是多输入数据总量
    Y是输出，单次输入X为k*1的向量
    多次输出Y为 k*m 的矩阵

    免神经层设计
    ****这里与myNN设计不同，采用WT替代W尝试：
    W是参数，k*n+1的矩阵，每列为各神经元参数，第一列是偏移

    [       ]       [       ]
    [   X   ]   --> [   Y   ]
    [       ]n*m    [       ]k*m

    """

    def __init__(self, num_input, outputLine: list):
        self.deep = len(outputLine)
        self.wList = np.ndarray((self.deep,), dtype=np.ndarray)
        for index in range(self.deep):
            if index == 0:
                # self.wList[index] = np.zeros((outputLine[index], num_input + 1), dtype=np.float64)
                self.wList[index] = np.random.random((outputLine[index], num_input + 1)) - 0.5
            else:
                # self.wList[index] = np.zeros((outputLine[index], outputLine[index - 1] + 1), dtype=np.float64)
                self.wList[index] = np.random.random((outputLine[index], outputLine[index - 1] + 1)) - 0.5

    def getResult(self, data_input):
        h = data_input
        for w in self.wList:
            h = sigmoid(w @ expandX(h))
        return h

    def fit(self, x, y, times=1000, eta=1e-1, pen=1e-1,
            batch_size=10,
            converge_gap=1e-10, diverge_gap=1e-2, acc_quit=99):
        self.lasterror = 1e5
        for time in range(times):
            try:
                for point in range(0, x.shape[1]-1, batch_size):
                    self.fit_batch(x[:, point:point+batch_size],
                                   y[:, point:point+batch_size], eta=eta, pen=pen)
            except Exception as e:
                print(e)
                self.fit_batch(x, y, eta=eta, pen=pen)
            if time % 10 == 0:
                eva = self.evaluate(x, y, pen=pen)
                print('times:', time)
                print('eva', eva)
                if acc_quit is None:
                    pass
                elif eva[0] > acc_quit:
                    print(f'Correctly! Acc over {acc_quit}%, quit fitting.')
                    return
                elif abs(eva[1] - self.lasterror) <= converge_gap:
                    print(f'Converged! by acc {eva[0]}')
                    return
                elif time > 10 and eva[1] - self.lasterror >= diverge_gap:
                    print(f'Diverged! by acc {eva[0]}')
                    return
                self.lasterror = eva[1]

    def fit_batch(self, x, y, eta=1e-5, pen=1e-1):
        m = y.shape[1]
        # forward propagation
        result = np.ndarray((self.deep,), dtype=np.ndarray)
        for index, w in enumerate(self.wList):
            if index == 0:
                result[index] = sigmoid(w @ expandX(x))
            else:
                result[index] = sigmoid(w @ expandX(result[index - 1]))
        # bp
        # error[l] = { y - r[L] , l = L , where r[l] is result of layer l , l = 1 .. L
        #            { W[l+1] * error[l+1] * df(l) , l != L
        # dw = { r[l-1] * error[l]' , where r is s(l-1) by 1 matrix , e is sl*1
        #      { x * e[l]'
        # its ok if x is matrix
        for i in range(self.deep).__reversed__():
            if i == self.deep - 1:
                delta = result[-1] - y
            else:
                delta = self.wList[i + 1].T @ delta * d(sigmoid)(expandX(result[i]))
                delta = delta[1:, :]  # remove delta0
            if i != 0:
                # expanded result : s(l-1)+1 * m , delta:sl * m
                # delta@result.T : sl * s(l-1)+1
                dw = 1 / m * delta @ expandX(result[i - 1]).T
            else:
                dw = 1 / m * delta @ expandX(x).T
            pen_matrix = pen * self.wList[i]
            pen_matrix[:, 1] = 0
            dw = dw + pen_matrix
            # gradient check , only print. close 'check' for speed
            # print(f"layer{i} BP:", dw)
            # print("CHECK:", self.gradientCheck(x, y, i, pen))
            self.wList[i] = self.wList[i] - eta * dw

    def gradientCheck(self, x, y, layer, pen):
        dw = np.ndarray(self.wList[layer].shape, dtype=np.float64)

        m = y.shape[1]
        step = 1e-4
        w_2 = np.ndarray((2,), dtype=np.ndarray)
        cost_2 = np.zeros((2,), dtype=np.float64)
        for i in range(self.wList[layer].shape[0]):
            for j in range(self.wList[layer].shape[1]):
                w_2[0] = self.wList[layer].copy()
                w_2[1] = self.wList[layer].copy()
                w_2[0][i, j] -= step / 2
                w_2[1][i, j] += step / 2
                for k in range(2):
                    h = x
                    for lay, w in enumerate(self.wList):
                        if lay != layer:
                            h = sigmoid(w @ expandX(h))
                        else:
                            h = sigmoid(w_2[k] @ expandX(h))
                    temp1 = -1 / m * np.sum(y * mylog(h) + (1 - y) * mylog(1 - h), axis=None)
                    temp2 = pen / 2 / m * sum([np.sum(w[1:, :] ** 2, axis=None) for w in self.wList])
                    cost_2[k] = temp1 + temp2
                dw[i, j] = (cost_2[1] - cost_2[0]) / step
        return dw

    def evaluate(self, x, y, pen):
        m = y.shape[1]
        h = x
        for w in self.wList:
            h = sigmoid(w @ expandX(h))
        eps = 1e-1
        acc = np.sum(np.abs(h - y) < eps, axis=None) / (y.shape[0] * y.shape[1]) * 100
        temp1 = -1 / m * np.sum(y * mylog(h) + (1 - y) * mylog(1 - h), axis=None)
        temp2 = pen / 2 / m * sum([np.sum(w[:, 1:] ** 2, axis=None) for w in self.wList])
        return acc, temp1 + temp2

    def fitAuto(self, x, y, pen=1e2):
        mininum = op.minimize(self.E, self.vectorizeWeight(self.wList), (x, y, pen), tol=1e-5, options={'disp': True})
        self.wList = self.reformatWeight(mininum.x)
        # mininum = op.fmin(self.E, self.vectorizeWeight(self.wList), (x, y, pen))
        # self.wList = self.reformatWeight(mininum)

    def E(self, wVector, x, y, pen=1e2):
        wList = self.reformatWeight(wVector)
        return self.E2(wList, x, y, pen=pen)

    def E2(self, wList, x, y, pen=1e2):
        m = y.shape[1]
        h = x
        for w in wList:
            h = sigmoid(w @ expandX(h))
        # y 是 k * m 的矩阵 ， h 是 k * m 的矩阵
        temp1 = -1 / m * np.sum(y * mylog(h) + (1 - y) * mylog(1 - h), axis=None)
        temp2 = pen / 2 / m * sum([np.sum(w[:, 1:] ** 2, axis=None) for w in wList])
        # for w in wList:
        #     temp2 += np.sum(w[:,1:]**2, axis=None)  # exclude col 0
        # temp2 *= pen/2/m
        # temp2 = pen / 2 / m * np.sum(wList ** 2, axis=None)
        return temp1 + temp2

    def vectorizeWeight(self, wList):
        length = sum([w.shape[0] * w.shape[1] for w in self.wList])
        vector = np.ndarray((length,), dtype=np.float64)
        pointer = 0
        for w in wList:
            vector[pointer: pointer + w.shape[0] * w.shape[1]] = w.reshape((w.shape[0] * w.shape[1],))
            pointer += w.shape[0] * w.shape[1]
        return vector

    def reformatWeight(self, vector):
        wList = np.ndarray((self.deep,), dtype=np.ndarray)
        pointer = 0
        for ind, w in enumerate(self.wList):
            wList[ind] = vector[pointer: pointer + w.shape[0] * w.shape[1]].reshape((w.shape[0], w.shape[1]))
            pointer += w.shape[0] * w.shape[1]
        return wList

    def show(self):
        """
        这可以展示网络的结构简图
        """
        rows = [''] * 5
        width = 12
        for index, w in enumerate(self.wList):
            rows[0] += '|' + "-" * (width - 2) + '|'
            row = "layer {}:".format(index + 1)
            rows[1] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            row = "input:{}".format(w.shape[1] - 1)
            rows[2] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            row = "output:{}".format(w.shape[0])
            rows[3] += '|' + row + (width - len(row) - 2) * ' ' + '|'
            rows[4] += '|' + "-" * (width - 2) + '|'
        print('|net:' + ' ' * (width * self.deep - 6) + '|')
        for row in rows:
            print(row)


def createNet():
    num_in, num_out = 3, 2
    out_list = [4, 5, num_out]
    net = Net(num_in, out_list)
    data_in = np.random.random((3, 1))
    print(net.getResult(data_in))
    net.show()


def vectorize():
    model = Net(3, [10, 10, 1])
    print(model.vectorizeWeight(model.wList))
    print(model.reformatWeight(model.vectorizeWeight(model.wList)))


def netFitAuto():
    model = Net(3, [10, 10, 1])
    data_in = np.asarray([[x, y, z] for x in range(2) for y in
                          range(2) for z in range(2)]).T
    data_out = np.asarray([[i for i in [1, 1, 1, 0, 1, 0, 0, 0]]])
    print(model.getResult(data_in))
    print(model.E2(model.wList, data_in, data_out, pen=1e-4))
    model.fitAuto(data_in, data_out, pen=1e-4)
    print(model.getResult(data_in))
    print(model.E2(model.wList, data_in, data_out, pen=1e-4))


def netfit():
    model = Net(3, [10, 10, 1])
    data_in = np.asarray([[x, y, z] for x in range(2) for y in
                          range(2) for z in range(2)]).T
    data_out = np.asarray([[i for i in [1, 1, 1, 0, 1, 0, 0, 0]]])
    args = {
        'times': 10000,
        'eta': 1e-1,
        'pen': 1e-2,
        'batch_size': 10,
        'acc_quit': 99
    }
    model.fit(data_in, data_out, **args)
    result = model.getResult(data_in)
    print(result)
    print(result >= 0.5)


if __name__ == '__main__':
    netfit()
