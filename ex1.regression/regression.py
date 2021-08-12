import numpy as np
import matplotlib.pyplot as plt


# 单变量线性回归
# 测试了半天原来是学习率太高，居然要压到1e-7才收敛
# update：
# 根据新学习的知识，应该采用参数放缩
#
class Model:
    """
    输入是1*1的向量，输出是标量
    预测模型 y = w0 + w1x1
    assume x0=1
    """

    def __init__(self):
        self.w0 = 0
        self.w1 = 1
        self.predict = np.frompyfunc(self.predict, 1, 1)

    def predict(self, x):
        return self.w0 + self.w1 * x

    def fit(self, data_in, data_out, eta=1e-5, error=100):
        i = 0
        m = len(data_out)
        self.lasterror = 1e30
        while True:
            i += 1
            temp1 = sum(self.predict(data_in) - data_out)
            temp2 = np.inner(self.predict(data_in) - data_out, data_in)
            self.w0 -= eta * 1 / m * temp1
            self.w1 -= eta * 1 / m * temp2
            temp2 = 1 / 2 / m * np.sum((self.predict(data_in) - data_out) ** 2)
            print('times: ', i, ' error: ', temp2)
            if abs(self.lasterror - temp2) < error:
                break
            if temp2 > 1e30:
                print('deverged!')
                break
            self.lasterror = temp2


def main():
    data_in = np.asarray([2104, 1416, 1534, 852])
    data_out = np.asarray([460, 232, 315, 178])

    model = Model()
    model.fit(data_in, data_out, eta=1e-7, error=1)
    plt.subplot(1, 2, 1)
    plt.plot(data_in, data_out)
    plt.subplot(1, 2, 2)
    x = np.arange(500, 3000, 100)
    y = model.predict(x)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
