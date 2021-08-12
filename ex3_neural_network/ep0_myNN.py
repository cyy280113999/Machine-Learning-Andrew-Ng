"""
my neural network

模型：
    神经元LinNeuron
    神经层Layer
    神经网Net

测试：
    createNeuron
    createLayer
    createNet
    showNet
    MatrixVersionTest
    updateLayer
    netfitTest 一个数据拟合
    netfit 多个数据拟合，不同网络设计测试
    batchTest
update:
    fit_batch with batchTest() more faster! sometimes diverged
"""
from myfunc import *
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class LinNeuron:
    """
    这是一个基本的神经元，是线性基函数，fun激活函数神经元
    单次输入，
    X是输入，n*1的向量，W是权重，n*1的向量，B是偏移，标量，
    输出Y=f(WT*X+B)，标量
    Y =[    WT    ]*[ ]+B
                    [X]
                    [ ]
    支持多输入，列向量合并成矩阵
    X是n*m的矩阵，Y是1*m的矩阵
    """
    init_mode = 'random'

    def __init__(self, num_input, fun=sigmoid):
        if self.init_mode == 'random':
            self.w = np.random.random((num_input, 1)) - 0.5
            self.b = np.random.random((1, 1)) - 0.5
        else:
            self.w = np.zeros((num_input, 1))
            self.b = np.zeros((1, 1))
        self.fun = fun

    def getResult(self, data_input):
        """
        :parameter data_input:X
        :return Y=f(WT*X+B)
        """
        # 多输入情况，左边行向量加右边常数实测可行
        return self.fun(self.w.T @ data_input + self.b)


def createNeuron():
    num_in = 3
    neu = LinNeuron(num_in)
    x = np.random.random((num_in, 1))
    print(neu.getResult(x))



class Layer:
    """
    这是神经层，元素为基本神经元
    X是输入，n*1的向量
    通过矩阵表示输出
    [ ]   [        ] [ ]                 [weight1|weight2|..|]       [b1]
    [Y] = [   WT   ]*[X]      where w is [       |       |  |] ,b is [b2]
    [ ]   [        ] [ ]                 [       |       |  |]       [b3]
    得到向量Y m*1
    网络的输入大小和每个神经元输入一致，输出大小是本层神经元的个数
    example：
              |---layer---|
              |    Neur   |
       input  |    Neur   |   output
              |    Neur   |
    支持多组输入
    X是n*k的矩阵，Y是m*k的矩阵
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
        self.neurList = np.ndarray((num_output,), dtype=LinNeuron)
        for i in range(num_output):
            self.neurList[i] = LinNeuron(num_input, self.fun)
        self.LayerW = np.ndarray((self.num_input, self.num_output), dtype=np.float64)
        self.LayerB = np.ndarray((self.num_output, 1), dtype=np.float64)
        self.getMatrix()

    def getResult(self, data_input):
        result = np.zeros((self.num_output, 1), dtype=np.float64)
        for index, neu in enumerate(self.neurList):
            result[index] = neu.getResult(data_input)
        return result

    def getMatrix(self):
        """更新矩阵表示的计算参数"""
        for i in range(self.num_output):
            self.LayerW[:, i] = self.neurList[i].w[:, 0]  # 这里因为numpy切片规则改变了矩阵形状，所以有些奇怪
            self.LayerB[i, 0] = self.neurList[i].b

    def getResultMatrixVersion(self, data_input):
        """
        矩阵表示多组数据计算方法
        :parameter data_input:X
        :return Y=f(WT*X+B)
        """
        # 多输入情况，左边矩阵加右边列向量可行
        return self.fun(self.LayerW.T @ data_input + self.LayerB)

    def update(self, dw, db):
        for index, neu in enumerate(self.neurList):
            neu.w[:, 0] = neu.w[:, 0] + dw[:, index]
            neu.b = neu.b + db[index]
        self.getMatrix()


def createLayer():
    num_in, num_out = 3, 2
    layer = Layer(num_in, num_out)
    data_in = np.random.random((num_in, 1))
    print(layer.getResult(data_in))


class Net:
    """
    这是多层神经网络，包含基本层
    X是输入，n*1的向量
    输出 Y = Ld(Ld-1(...(L1(X)...)), exp:L1(X) = sigmod(WT*X+B)
    得到向量Y m*1

    要生成网络，请给出X的长度，和包含隐含层的神经元个数的列表，列表的长度决定网络的层数
    例：3,[5,6,7]表示三层神经网络，输入X 3*1，输出Y 7*1

    支持BP学习规则

    支持多组输入
    X是n*k的矩阵，Y是m*k的矩阵
    """

    def __init__(self, num_input, outputLine):
        """
        :param num_input: 输入的尺寸
        :param outputLine: 各层的尺寸，最后一层的大小即为输出尺寸
        update：修改为激活函数固定
        """
        self.deep = len(outputLine)
        # self.fun = fun
        self.layerList = np.ndarray((self.deep,), dtype=Layer)
        for index in range(self.deep):
            if index == 0:
                self.layerList[index] = Layer(num_input, outputLine[index], sigmoid)
            elif index == self.deep - 1:
                self.layerList[index] = Layer(outputLine[index - 1], outputLine[index], sigmoid)
            else:
                self.layerList[index] = Layer(outputLine[index - 1], outputLine[index], sigmoid)

    def getResult(self, data_input):
        result = data_input
        for layer in self.layerList:
            result = layer.getResult(result)
        return result

    def getResults(self, set_input):
        result = [None] * len(set_input)
        for index, datain in enumerate(set_input):
            result[index] = self.getResult(datain)
        return result

    def getResultMatrixVersion(self, data_input):
        result = data_input
        for layer in self.layerList:
            result = layer.getResultMatrixVersion(result)
        return result

    def fit_once(self, data_input: np.ndarray, data_output: np.ndarray, eta=1e-5):
        """
        梯度下降法 误差反向传播 学习规则
        参数更新：w = w + Δw , where Δw is - η * ∇E(w), where E(w) is squared error function, eta is learning rate.
        """
        # 计算并保留各层输出
        result = [None] * self.deep
        for index, layer in enumerate(self.layerList):
            if index == 0:
                result[index] = layer.getResult(data_input)
            else:
                result[index] = layer.getResult(result[index - 1])
        # 反向计算更新权值
        for i in range(self.deep).__reversed__():
            # update : add simplified code to comment
            # alternatively,
            # error[l] = { y - r[L] , l = L , where r[l] is result of layer l , l = 1 .. L
            #            { W[l+1] * error[l+1] * df(l) , l != L
            # dw = { r[l-1] * error[l]' , where r is s(l-1) by 1 matrix , e is sl*1
            #      { x * e[l]'
            # 如果是倒数第一层，则
            if i == self.deep - 1:
                error = data_output - result[-1]
            # 其他层的局部误差
            else:
                error = self.layerList[i + 1].LayerW @ delta  # w is  sl*s(l+1) , delta is s(l+1)*1 , error is sl*1
            # 局部梯度
            delta = error * d(self.layerList[i].fun)(result[i])  # result[i] is sl*1 , delta is sl*1
            # 更新权值
            if i != 0:
                dw = eta * result[i - 1] @ delta.T  # result[i-1] is s(l-1)*1 , dw is s(l-1)*sl
            # 最后一层不同：
            else:
                dw = eta * data_input @ delta.T
            db = eta * delta
            self.layerList[i].update(dw, db)
        return None

    def fit_batch(self, data_input: np.ndarray, data_output: np.ndarray, eta=1e-5):
        # fp
        result = np.ndarray((self.deep,), dtype=np.ndarray)
        for index, layer in enumerate(self.layerList):
            if index == 0:
                result[index] = layer.getResultMatrixVersion(data_input)
            else:
                result[index] = layer.getResultMatrixVersion(result[index - 1])
        # bp
        for i in range(self.deep).__reversed__():
            if i == self.deep - 1:
                delta = data_output - result[-1]
            else:
                delta = self.layerList[i + 1].LayerW @ delta * d(self.layerList[i].fun)(result[i])
            if i != 0:
                # result : s(l-1)*m , delta:sl*m
                # result@deltaT : s(l-1)*sl
                dw = eta * result[i - 1] @ delta.T
            else:
                dw = eta * data_input @ delta.T
            db = eta * np.sum(delta, axis=1)
            self.layerList[i].update(dw, db)
        return None

    # input data set(list)
    def fitSet(self, setin, setout, eta=1e-5, each=10, times=100):
        try:
            self.lasterror = 1e5
            for i in range(times):
                for (datain, dataout) in zip(setin, setout):
                    for j in range(each):
                        self.fit_once(datain, dataout, eta=eta)
                if i % 10 == 0:
                    eva = self.evaluate(setin, setout)
                    print('times:', i)
                    print('eva', eva)
                    if eva[1] > self.lasterror:
                        print('deverged!')
                        return
                    self.lasterror = eva[1]
        except Exception as e:
            print(e)

    # input matrix
    def fitMatrix(self, data_in, data_out, eta=1e-5, each=10, epoch=100):
        try:
            self.lasterror = 1e5
            data_len = data_in.shape[1]
            for time in range(epoch):
                for i in range(data_len):
                    for ii in range(each):
                        self.fit_once(data_in[:, i, None], data_out[:, i, None], eta=eta)
                if time % 10 == 0:
                    eva = self.evaluate(data_in, data_out)
                    print('times:', time)
                    print('eva', eva)
                    if eva[0] > 99:
                        print('Converged.')
                        return
                    if eva[1] > self.lasterror + 1:
                        print('deverged!')
                        return
                    self.lasterror = eva[1]
        except Exception as e:
            print(e)
            raise

    # batch version
    def fit(self, data_in, data_out, eta=1e-5, epoch=100):
        converge_gap = 1e-3
        self.lasterror = 1e5
        for time in range(epoch):
            self.fit_batch(data_in, data_out, eta=eta)
            if time % 10 == 0:
                eva = self.evaluate(data_in, data_out)
                print('times:', time)
                print('eva', eva)
                if eva[0] > 99:
                    print('Correctly! Acc over 99%, quit fitting.')
                    return
                elif abs(eva[1] - self.lasterror) <= converge_gap:
                    print(f'Converged! by acc {eva[0]}')
                    return
                self.lasterror = eva[1]


    def evaluate(self, data_in, data_out):
        eps = 1e-1
        result = self.getResultMatrixVersion(data_in)
        acc = np.sum(np.abs(result - data_out) < eps, axis=None) / (data_out.shape[0] * data_out.shape[1]) * 100
        return acc, 1 / 2 / len(data_out) * np.sum((result - data_out) ** 2)

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
        print('|ex3_neural_network:' + ' ' * (width * len(self.layerList) - 6) + '|')
        for row in rows:
            print(row)


def createNet():
    num_in, num_out = 3, 2
    out_list = [4, 5, num_out]
    net = Net(num_in, out_list)
    data_in = np.random.random((3, 1))
    print(net.getResult(data_in))


def showNet():
    net = Net(5, [6, 7, 8, 3])
    net.show()


def MatrixVersionTest():
    layer = Layer(5, 5)
    data = np.random.random((5, 1))
    a = layer.getResult(data)
    b = layer.getResultMatrixVersion(data)
    print('a is ', a)
    print('b is ', b)
    print('a==b?', a == b)
    net = Net(3, [5, 6, 3])
    data = np.random.random((3, 1))
    a = net.getResult(data)
    b = net.getResultMatrixVersion(data)
    print('a is ', a)
    print('b is ', b)
    print('a==b?', a == b)


def updateLayer():
    layer = Layer(5, 2)
    data = np.random.random((5, 1))
    layer.getResultMatrixVersion(data)
    print(layer.LayerW)
    print(layer.LayerB)
    w = np.random.random((5, 2))
    b = np.random.random((2, 1))
    layer.update(w, b)
    layer.getResultMatrixVersion(data)
    print(layer.LayerW)
    print(layer.LayerB)




def netfitTest():
    net = Net(3, [10, 10, 10, 3])
    # datai = n.random.random((3, 1))
    # datao = n.random.random((3, 1))
    datais = [np.asarray([x, y, z]).reshape(3, 1) for x in range(2) for y in
              range(2) for z in range(2)]
    dataos = [np.asarray([i, i, i]).reshape(3, 1) for i in [0, 1, 0, 1, 0, 1, 0, 1]]
    print('no fit:')
    print(net.getResults(datais))
    for i in range(1000):
        for j in range(8):
            net.fit_once(datais[j], dataos[j])
    print('fited:')
    print(net.getResults(datais))


def netfit():
    # 网络设计方法
    # 可层数多：Net(3, [10, 10, 10, 1])
    # or 神经元多：ex3_neural_network = Net(3, [100, 100, 1])
    # 可学习效率：eta=1e-1  eta=1e-7

    # data_in = [n.asarray([x, y, z]).reshape(3, 1) for x in range(2) for y in
    #           range(2) for z in range(2)]
    # data_in = [n.asarray([1, 1, 1]).reshape((3, 1))]
    # data_out = [n.asarray([0]).reshape((1, 1))]

    # 以下是一次可以成功运行的示例：反向三人表决器

    # out put function = sigmod , other layer = fone
    # net = Net(3, [100, 100, 1])
    # data_in = np.asarray([[x, y, z] for x in range(2) for y in
    #                       range(2) for z in range(2)]).T
    # data_out = np.asarray([[i] for i in [1, 1, 1, 0, 1, 0, 0, 0]]).T
    # net.fitMatrix(data_in, data_out, eta=1e-5, each=10, epoch=1000)

    # all layer = sigmod
    net = Net(3, [10, 10, 1])
    data_in = np.asarray([[x, y, z] for x in range(2) for y in
                          range(2) for z in range(2)]).T
    data_out = np.asarray([[i for i in [1, 1, 1, 0, 1, 0, 0, 0]]])
    # net.fitMatrix(data_in, data_out, eta=1e-1, each=10, epoch=1000)

    print('data in:')
    print(data_in)
    print('expect out:')
    print(data_out)
    print('no fit:')
    print(net.getResultMatrixVersion(data_in))
    net.fitMatrix(data_in, data_out, eta=1e-1, each=10, epoch=1000)
    print('fitted:')
    result = net.getResultMatrixVersion(data_in)
    print(result)

def batchTest():
    net = Net(3, [10, 10, 1])
    data_in = np.asarray([[x, y, z] for x in range(2) for y in
                          range(2) for z in range(2)]).T
    data_out = np.asarray([[i for i in [1, 1, 1, 0, 1, 0, 0, 0]]])
    print(net.getResultMatrixVersion(data_in))
    net.fit(data_in, data_out, eta=1e-1, epoch=10000)  # new version faster!
    print(net.getResultMatrixVersion(data_in))

if __name__ == '__main__':
    batchTest()
