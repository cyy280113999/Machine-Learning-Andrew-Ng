# unable to name regularization.py.        ?????????
"""
regularization of linear regression
正则化方法用于回归

函数过拟合：
数据参数多余，特征无用
模型参数复杂，多项式阶数高

函数过拟合的处理方法：
去掉多余数据参数
正则化，加惩罚使参数小

其实没用在线性上
模型：多项式回归FMinModel
    正则化多项式回归RModel抑制高阶项
分类：
    多项式回归fminTest
    正则化多项式回归regularization_pr
可以看到正则化前，曲线x域内四个拐点，正则化后，无拐点
"""

import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt


# 多项式回归 库搜索拟合
# 差别在 E（）
class FMinModel:
    """
    预测模型 y = w0 + w1x1 + w2*x1^2 + w3*x1^3 + w4*x1^4  , by order 4
    """

    def __init__(self, order):
        self.order = order
        self.w = np.random.random((self.order + 1, 1))

    def predict(self, x):
        x_vector = np.ones((self.order + 1, 1)) * x
        x_vector[0, :] = 1
        x_vector = np.cumprod(x_vector, axis=0)
        return self.w.T @ x_vector

    def fit(self, x, y):
        # minimum = op.fmin(self.E2, self.w, (x, y), ftol=1e-5, full_output=True) # works not well
        # w = minimum[0]
        minimum = op.minimize(self.E2, self.w.flatten(), (x, y), tol=1e-3, options={'disp': True})
        w = minimum.x
        self.w = w.reshape(self.w.shape)

    # def E(self, x, y):
    #     """
    #     误差函数为均方差 E = 1/2/len()*sum((h(x)-y)^2)
    #     """
    #     m = len(x)
    #     return 1 / 2 / m * np.sum(np.square(self.predict(x) - y))

    def E2(self, w, x, y):
        w = w.reshape(self.w.shape)
        x_vector = np.ones((self.order + 1, 1)) * x
        x_vector[0, :] = 1
        x_vector = np.cumprod(x_vector, axis=0)
        m = len(x)
        return 1 / 2 / m * np.sum(np.square(w.T @ x_vector - y))


# 正则化多项式回归 polynomial regression
class RModel:
    """
    四阶回归预测模型 y = w0 + w1x1 + w2*x1^2 + w3*x1^3 + w4*x1^4
    """

    def __init__(self, order):
        self.order = order
        self.w = np.zeros((self.order + 1, 1), dtype=np.float64)

    def predict(self, x):
        x_vector = np.ones((self.order + 1, 1), dtype=np.float64) * x
        x_vector[0, :] = 1
        x_vector = np.cumprod(x_vector, axis=0)
        return self.w.T @ x_vector

    def fit(self, x, y, pen=1e3):
        # minimum = op.fmin(self.E2, self.w.flatten(), (x, y), ftol=1e-5, full_output=True)
        minimum = op.minimize(self.E2, self.w.flatten(), (x, y, pen), tol=1e-3, options={'disp': True})
        w, E = minimum.x, minimum.fun
        # print('error:', E)
        self.w = w.reshape(self.w.shape)
        return

    # def E(self, x, y, pen=1e3):
    #     """
    #     误差函数为均方差 E = 1/2/len()*sum((h(x)-y)^2)
    #     """
    #     m = len(x)
    #     return 1 / 2 / m * np.sum(np.square(self.predict(x) - y)) \
    #            + pen * np.sum(np.square(self.w[2:]))

    def E2(self, w, x, y, pen=1e3):
        w = w.reshape(self.w.shape)
        x_vector = np.ones((self.order + 1, 1), dtype=np.float64) * x
        x_vector[0, :] = 1
        x_vector = np.cumprod(x_vector, axis=0)
        m = len(x)
        # 参数取平方作为惩罚
        return 1 / 2 / m * np.sum(np.square(w.T @ x_vector - y)) \
               + pen * np.sum(np.square(w[2:]))


def fminTest():
    # 可以看到 数据集内外 确实有过拟合

    # model = FMinModel(5)
    # x = 2
    # print(model.predict(x))

    # x = np.asarray([[1, 2, 2.7, 5, 7, 8, 9]])  # len = 7
    # x_range = np.linspace(-5, 15, 100)
    # y = np.asarray([[1.3, 4, 5, 6, 6.2, 6.5, 6.6]])
    # 含噪声的数据
    x = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.float64)
    x_range = np.linspace(-5, 15, 100, dtype=np.float64)
    y = np.asarray([[0, 2, 4, 3.5, 4.5, 6.8, 7.5, 7.3, 7.5, 8.1]], dtype=np.float64)

    # plt.subplot(1, 3, 1);plt.xlim(0, 10); plt.ylim(0, 10);plt.title('raw data')
    # plt.plot(x.flatten(), y.flatten())
    # print(model.predict(x))
    # print(model.E(x, y))

    # plt.subplot(1, 3, 2);plt.xlim(0, 10); plt.ylim(0, 10);plt.title('no fit')
    # plt.plot(x_range.flatten(), model.predict(x_range).flatten())
    for i in range(1, 7):
        model = FMinModel(5)  # order = 5
        model.fit(x, y)
        # print(model.predict(x))
        # print(model.E(x, y))
        plt.subplot(2, 3, i), plt.xlim(-5, 15), plt.ylim(-5, 15)
        plt.title('fitted')
        plt.plot(x.flatten(), y.flatten(), 'ro', label='raw')
        plt.plot(x_range.flatten(), model.predict(x_range).flatten(), label='fitted')
        plt.legend()
    plt.show()


def regularization_pr():
    # x = np.asarray([[1, 2, 2.7, 5, 7, 8, 9]], dtype=np.float64)
    # x_range = np.linspace(-5, 15, 100, dtype=np.float64)
    # y = np.asarray([[1.3, 4, 5, 6, 6.2, 6.5, 6.6]], dtype=np.float64)
    # 含噪声的数据
    x = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.float64)
    x_range = np.linspace(-5, 15, 100, dtype=np.float64)
    y = np.asarray([[0, 2, 4, 3.5, 4.5, 6.8, 7.5, 7.3, 7.5, 8.1]], dtype=np.float64)

    for i in range(1, 7):
        model = RModel(5)
        model.fit(x, y, 1e1)  # 10 enough
        # print(model.predict(x))
        # print(model.E(x, y))
        plt.subplot(2, 3, i), plt.xlim(-5, 15), plt.ylim(-5, 15)
        plt.title('fitted')
        plt.plot(x.flatten(), y.flatten(), 'ro', label='raw')
        plt.plot(x_range.flatten(), model.predict(x_range).flatten(), label='fitted')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    regularization_pr()
