"""
logistic回归
模型：线性logistic回归LRModel
    非线性logistic回归NonLinearLRModel
分类：
    线性一元train1
    线性二元train2para2
    非线性二元nonLinearClassification
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def sigmoid(data):
    # result = (1 - n.exp(-lam * data)) / (1 + n.exp(-lam * data))
    if data >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0 / (1 + np.exp(-data))
    else:
        return np.exp(data) / (1 + np.exp(data))
    # return 1 / (1 + n.exp(-lam * data))


sigmoid = np.frompyfunc(sigmoid, 1, 1)


def expandpara(para):
    data = np.ndarray(shape=(para.shape[0] + 1, para.shape[1]))
    data[0, :] = np.ones((1, data.shape[1]))
    data[1:, :] = para
    return data


class LRModel:
    """
    logistics model
    s折线回归，用来实现分类

    输入是n*1的向量，输出是二值，取[0,1]
    预测模型 y = g(w0 + w1x1 + w2x2 + ... + w4x4)>=0.5
    g是 sigmoid function, or logistic function
    """

    def __init__(self, num_in):
        self.w = np.zeros((num_in + 1, 1))

    def predict(self, data_in):
        """
        单次输入列向量，输出分类判断0 or 1
        多次输入为矩阵的各列，输出为行向量
        """
        data = expandpara(data_in)
        return np.asarray(sigmoid(self.w.T @ data) >= 0.5, dtype=int)

    def fit(self, data_in, data_out, eta=1e-3, times=1000):
        """
        the cost function before is E = 1/2/m*sum((h(x)-y)^2)
        we changed it into E = 1/m*sum(Cost(h(x),y)), Cost(h(x),y)= 1/2*(h(x)-y)^2
        today we replace the Cost():
                Cost(h(x),y)={-log(h(x)) ,y=1
                             {-log(1-h(x)),y=0
        equally    Cost(h(x),y) = -y*log(h(x))-(1-y)log(1-h(x))
        learning algorithm：
            w = w + Δw , where Δw is - η * ∇E(w), where E(w) is squared error function, eta is learning rate.
        ∇E(w) = 1/len()*sum((predict(data_in)-data_out))*x
        """
        self.lasterror = 1e5
        i = 0
        data = expandpara(data_in)
        while True:
            i += 1
            # w -= eta * 1 / m * (x @ (h(w.T @ x) - y).T)
            m = data_out.shape[1]
            # 解决log(0)为无穷的问题
            error_max = 1e5
            mylog = lambda x: np.log(x) if x != 0 else error_max
            mylog = np.frompyfunc(mylog, 1, 1)
            h = sigmoid(self.w.T @ data)
            temp1 = - data_out @ mylog(h).T
            temp2 = - (1 - data_out) @ mylog(1 - h).T
            self.w = self.w - eta * 1 / m * (data @ (h - data_out).T)
            temp2 = 1 / m * (temp1 + temp2)
            print('times: ', i, ' error: ', temp2)
            if i >= times:
                break
            if temp2 > 1e5:
                print('deverged!')
                break
            self.lasterror = temp2


def plotsigmod():
    x = np.linspace(-10, 10, 50)
    y = sigmoid(x)
    plt.plot(x, y, 'ro')
    plt.show()


def export():
    model = LRModel(4)
    data_in = np.random.random((4, 1))
    print(model.predict(data_in))


def train1():
    model = LRModel(1)
    data_in = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]])
    data_out = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1]])
    print('before:\n', model.predict(data_in))
    model.fit(data_in, data_out, eta=1e-2, times=1e4)
    print('after:\n', model.predict(data_in))


def train2para1():
    # failed in circle source
    model = LRModel(2)
    generate_function = lambda x, y: 1 if x ** 2 + y ** 2 >= 1 else 0
    generate_function = np.frompyfunc(generate_function, 2, 1)
    x_range = np.linspace(-2, 2, 10)
    y_range = np.linspace(-2, 2, 10)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_range = np.ravel(x_grid)
    y_range = np.ravel(y_grid)
    data_in = np.array([x_range, y_range])
    data_out = np.array([generate_function(x_range, y_range)])
    print('before:\n', model.predict(data_in))
    model.fit(data_in, data_out, eta=1e-1, times=1000)
    print('after:\n', model.predict(data_in))


def train2para2():
    model = LRModel(2)
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

    model.fit(data_in, data_out, eta=1e-2, times=100)

    plt.subplot(1, 3, 3)
    pred_out = model.predict(data_in)
    temp1 = data_in * pred_out
    temp2 = data_in * (1 - pred_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    plt.show()


error_max = 1e5
mylog = lambda para: np.log(para) if para != 0 else error_max
mylog = np.frompyfunc(mylog, 1, 1)


# 非线性模型 用来解决 圆边界分类
class NonLinearLRModel:
    """
    二元二次 logistic 回归
    预测模型 y = g(w0 + w1x1 + w2x1^2 + w3x2 + w4x2^2)>=0.5
    """

    def __init__(self):
        # np.asarray([-1, 0, 1, 0, 1]) is best
        self.w = 2 * np.random.random((5,)) - 1

    def predict(self, data_in):
        return np.asarray(sigmoid(self.w[0]
                                  + self.w[1] * data_in[0, :]
                                  + self.w[2] * data_in[0, :] ** 2
                                  + self.w[3] * data_in[1, :]
                                  + self.w[4] * data_in[1, :] ** 2
                                  ) >= 0.5, dtype=int)

    def fit(self, data_in, data_out):
        # minimum = op.fmin(self.E, self.w, (data_in, data_out), ftol=1e-5,
        #                   full_output=True)  # works not well
        # w = minimum[0]
        minimum = op.minimize(self.E, self.w, (data_in, data_out), tol=1e-5, options={'disp': True})
        w = minimum.x
        self.w = w

    def E(self, w, x, y):
        m = x.shape[1]
        h = sigmoid(w[0]
                    + w[1] * x[0, :]
                    + w[2] * x[0, :] ** 2
                    + w[3] * x[1, :]
                    + w[4] * x[1, :] ** 2
                    )
        return 1 / m * (- y @ mylog(h).T - (1 - y) @ mylog(1 - h).T)


def nonLinearClassification():
    # 成功
    model = NonLinearLRModel()
    generate_function = lambda x, y: 1 if x ** 2 + y ** 2 >= 1 else 0
    generate_function = np.frompyfunc(generate_function, 2, 1)
    x_range = np.linspace(-1.5, 1.5, 20)
    y_range = np.linspace(-1.5, 1.5, 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    x_range = np.ravel(x_grid)
    y_range = np.ravel(y_grid)
    data_in = np.array([x_range, y_range])
    data_out = np.array([generate_function(x_range, y_range)])

    plt.subplot(1, 3, 1), plt.title('raw input')
    temp1 = data_in * data_out
    temp2 = data_in * (1 - data_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    # print('before:\n', model.predict(data_in))
    plt.subplot(1, 3, 2), plt.title('no fit')
    pred_out = model.predict(data_in)
    temp1 = data_in * pred_out
    temp2 = data_in * (1 - pred_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    model.fit(data_in, data_out)
    # print('after:\n', model.predict(data_in))
    plt.subplot(1, 3, 3), plt.title('fitted')
    pred_out = model.predict(data_in)
    temp1 = data_in * pred_out
    temp2 = data_in * (1 - pred_out)
    plt.plot(temp1[0, :], temp1[1, :], 'bo', temp2[0, :], temp2[1, :], 'ro')

    plt.show()


if __name__ == '__main__':
    nonLinearClassification()
