import numpy as np
import matplotlib.pyplot as plt


# 多元线性回归
class MLModel:
    """
    输入是n*1的向量，输出是标量
    预测模型 y = w0 + w1x1 + w2x2 + ... + w4x4
    assume x0=1
    """

    def __init__(self, num_in):
        self.w = np.zeros((num_in + 1, 1))

    def predict(self, data_in):
        """
        单次输入列向量，输出标量1*1
        多次输入为矩阵的各列，输出为行向量
        """
        shape = list(data_in.shape)
        shape[0] += 1
        data = np.ndarray(shape=shape)
        data[0, :] = np.ones((1, shape[1]))
        data[1:, :] = data_in
        return self.w.T @ data

    def fit(self, data_in, data_out, eta=1e-5, error=100):
        """
        误差函数为均方差 E = 1/2/len()*sum((predict(data_in)-data_out)^2)
        参数更新：w = w + Δw , where Δw is - η * ∇E(w), where E(w) is squared error function, eta is learning rate.
        ∇E(w) = 1/len()*sum((predict(data_in)-data_out))*x
        """
        self.lasterror = 1e30
        i = 0
        # create data
        shape = list(data_in.shape)
        shape[0] += 1
        data = np.ndarray(shape=shape)
        data[0, :] = np.ones((1, shape[1]))
        data[1:, :] = data_in
        #
        while True:
            i += 1
            # temp1 = np.zeros(self.w.shape)
            # for col in range(data_in.shape[1]):
            #     temp1 += (self.w.T @ data[:, col, None] - data_out[0, col]) * data[:, col, None]
            # self.w -= eta * 1 / len(data_out) * temp1
            # w -= eta * 1 / m * (x @ (w.T @ x - y).T) which is vectorized.
            self.w -= eta * 1 / len(data_out) * (data @ (self.w.T @ data - data_out).T)
            temp2 = 1 / 2 / len(data_out) * np.sum((self.w.T @ data - data_out) ** 2)
            print('times: ', i, ' error: ', temp2)
            if abs(self.lasterror - temp2) < error:
                break
            if temp2 > 1e30:
                print('deverged!')
                break
            self.lasterror = temp2

    def E(self, data_in, data_out):
        return 1 / 2 / len(data_out) * np.sum((self.predict(data_in) - data_out) ** 2)




def matirxTest():
    data = np.random.random((4, 4))
    # print(data)
    # for row in data:
    #     print(row)
    # 先按行抽取
    # print(data[0:1,:].shape)
    # print(data[None,0,:].shape)
    # +None 避免维数缺失
    a = np.asarray([2104, 5, 1, 45]).reshape((4, 1))
    data[:, 0, None] = a
    print(data[:, 0, None])


def main():
    # 输入4，输出1，4组数据
    data_in = np.ndarray((4, 4))
    data_out = np.ndarray((1, 4))
    data_in[None, :, 0] = np.asarray([2104, 5, 1, 45])
    data_in[None, :, 1] = np.asarray([1416, 3, 2, 40])
    data_in[None, :, 2] = np.asarray([1534, 3, 2, 30])
    data_in[None, :, 3] = np.asarray([852, 2, 1, 36])
    data_out[0, :] = np.asarray([[460, 232, 315, 178]])

    model = MLModel(4)
    model.fit(data_in, data_out, eta=1e-8, error=1)
    plt.plot(range(4), data_out[0, :], 'r')
    plt.plot(range(4), model.predict(data_in)[0, :], 'g')
    plt.show()


if __name__ == '__main__':
    main()
