"""
模型：
    SVM

测试：
    costTest
    LinearClassify
    nonLinearClassification
update：
    创建预测predict
    创建损失函数cost
    创建优化函数fit
    创建了核函数版本SVMF
    创建了特征学习方法
"""
from myfunc import *
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification


class SVM:
    def __init__(self, num_input):
        self.num_input = num_input
        self.theta = 2 * np.random.random((1, self.num_input + 1)) - 1

    def predict(self, x):
        return np.asarray(self.theta @ expandX(x) >= 0, dtype=int)

    def fit(self, x, y):
        mininum = op.minimize(self.cost, self.theta.flatten(), (x, y), tol=1e-5, options={'disp': True})
        self.theta = mininum.x.reshape(self.theta.shape)
        pass

    def cost1(self, theta, x):
        # vectorized
        val = theta @ expandX(x)
        result = -6 * (val - 1)
        result[val >= 1] = 0
        return result

    def cost0(self, theta, x):
        val = theta @ expandX(x)
        result = 6 * (val + 1)
        result[val <= -1] = 0
        return result

    def cost(self, theta, x, y, c=1e3):
        theta = theta.reshape(self.theta.shape)
        A = np.sum(y * self.cost1(theta, x) + (1 - y) * self.cost0(theta, x) + 1 / 2)
        B = np.sum(self.theta[:, 1:] ** 2)
        return c * A + B


def costTest():
    model = SVM(2)
    generate_function = lambda x, y: 1 if x + y >= 3 else 0
    generate_function = np.frompyfunc(generate_function, 2, 1)
    x_range = np.linspace(0, 3, 10)
    y_range = np.linspace(0, 3, 10)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_range = np.ravel(x_grid)
    y_range = np.ravel(y_grid)
    data_in = np.array([x_range, y_range])
    data_out = np.array([generate_function(x_range, y_range)], dtype=np.int)
    print(model.cost(model.theta, data_in, data_out, 1e3))


def LinearClassify():
    model = SVM(2)
    generate_function = lambda x, y: 1 if x + y >= 3 else 0
    generate_function = np.frompyfunc(generate_function, 2, 1)
    x_range = np.linspace(0, 3, 10)
    y_range = np.linspace(0, 3, 10)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_range = np.ravel(x_grid)
    y_range = np.ravel(y_grid)
    data_in = np.array([x_range, y_range])
    data_out = np.array([generate_function(x_range, y_range)], dtype=np.int)

    plt.subplot(1, 3, 1)
    temp1 = data_in * data_out
    temp2 = data_in * (1 - data_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    plt.subplot(1, 3, 2)
    pred_out = model.predict(data_in)
    temp1 = data_in * pred_out
    temp2 = data_in * (1 - pred_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    model.fit(data_in, data_out)

    plt.subplot(1, 3, 3)
    pred_out = model.predict(data_in)
    temp1 = data_in * pred_out
    temp2 = data_in * (1 - pred_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    plt.show()


class SVMF:
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.theta = 2 * np.random.random((1, feature_num + 1)) - 1

    def predict(self, x):
        return np.asarray(self.theta @ expandX(self.getFeature(x)) >= 0, dtype=int)

    def fit(self, x, y, c=1e3, sig2=1e-1):
        if x.shape[1] != self.feature_num:
            raise Exception('sample num not complicate')
        self.landmark = x
        self.sig2 = sig2
        f = self.getFeature(x)
        mininum = op.minimize(self.cost, self.theta.flatten(), (f, y, c), tol=1e-5, options={'disp': True})
        self.theta = mininum.x.reshape(self.theta.shape)

    def getFeature(self, x):
        f = np.ndarray((self.feature_num, x.shape[1]))
        # f is not expanded
        # for f vector example of input x , f1 = gs(x,x1)
        # for xi , fij = gs(xi,xj)
        for col in range(f.shape[1]):
            for row in range(f.shape[0]):
                f[row, col] = gaussian(x[:, col], self.landmark[:, row], self.sig2)
        return f

    def cost1(self, theta, x):
        # vectorized
        val = theta @ expandX(x)
        result = -6 * (val - 1)
        result[val >= 1] = 0
        return result

    def cost0(self, theta, x):
        val = theta @ expandX(x)
        result = 6 * (val + 1)
        result[val <= -1] = 0
        return result

    def cost(self, theta, x, y, c=1e3):
        theta = theta.reshape(self.theta.shape)
        A = np.sum(y * self.cost1(theta, x) + (1 - y) * self.cost0(theta, x) + 1 / 2)
        B = np.sum(self.theta[:, 1:] ** 2)
        return c * A + B


def nonLinearClassification():
    model = SVMF(100)
    # generate_function = lambda x, y: 1 if x ** 2 + y ** 2 >= 1 else 0
    # generate_function = np.frompyfunc(generate_function, 2, 1)
    # x_range = np.linspace(-1.5, 1.5, 10)
    # y_range = np.linspace(-1.5, 1.5, 10)
    # x_grid, y_grid = np.meshgrid(x_range, y_range)
    # x_range = np.ravel(x_grid)
    # y_range = np.ravel(y_grid)
    # data_in = np.array([x_range, y_range])
    # data_out = np.array([generate_function(x_range, y_range)])

    # 中心分布数据
    # noise 影响边界模糊 ， factor 影响边界距离
    X, labels = make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=1)

    # 边界分布数据
    # X, labels = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    #                                 random_state=1, n_clusters_per_class=2)
    data_in = X.T
    data_out = labels.reshape((1, 100))

    xL, xR = -2, 2
    yL, yR = -2, 2

    plt.subplot(1, 3, 1), plt.title('raw input')
    plt.xlim(xL, xR), plt.ylim(xL, xR)
    Positive_samples = data_in[:, data_out[0, :] == 1]
    Negative_samples = data_in[:, data_out[0, :] == 0]
    plt.plot(Positive_samples[0, :], Positive_samples[1, :], 'bo', Negative_samples[0, :], Negative_samples[1, :], 'ro')

    model.fit(data_in, data_out,c=1e1)
    plt.subplot(1, 3, 2), plt.title('fitted')
    plt.xlim(xL, xR), plt.ylim(xL, xR)
    pred_out = model.predict(data_in)
    Positive_samples = data_in[:, pred_out[0, :] == 1]
    Negative_samples = data_in[:, pred_out[0, :] == 0]
    plt.plot(Positive_samples[0, :], Positive_samples[1, :], 'bo', Negative_samples[0, :], Negative_samples[1, :], 'ro')

    plt.subplot(1, 3, 3), plt.title('scanned')
    x_range = np.linspace(xL, xR, 100)
    y_range = np.linspace(xL, xR, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_range = np.ravel(x_grid)
    y_range = np.ravel(y_grid)
    data_in = np.array([x_range, y_range])
    pred_out = model.predict(data_in)
    Positive_samples = data_in[:, pred_out[0, :] == 1]
    Negative_samples = data_in[:, pred_out[0, :] == 0]
    plt.plot(Positive_samples[0, :], Positive_samples[1, :], 'bo', Negative_samples[0, :], Negative_samples[1, :], 'ro')
    plt.show()


if __name__ == '__main__':
    nonLinearClassification()
