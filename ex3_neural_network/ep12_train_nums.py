"""
mnist手写数字数据集训练ep1模型测试

模型：
    ThisNet 继承了ep1中的 Net
            改写了fit方法

"""
from ep1_nn import *
import tensorflow.keras.datasets.mnist as mn


class ThisNet(Net):
    def fit(self, x, y, x_t, y_t, times=1000, eta=1e-1, pen=1e-1,
            batch_size=10,
            converge_gap=1e-10, diverge_gap=1e-2, acc_quit=99):
        # 打印初始状态
        eva = self.evaluate(x, y, pen=pen)
        print('start:')
        print('eva', eva)

        lasterror = 1e5
        # 记录每次学习的准确度和误差，用来画图
        tr_error = np.zeros((times,))
        tr_acc = np.zeros((times,))
        te_error = np.zeros((times,))
        te_acc = np.zeros((times,))

        for time in range(times):
            try:
                for point in range(0, x.shape[1] - 1, batch_size):
                    self.fit_batch(x[:, point:point + batch_size],
                                   y[:, point:point + batch_size], eta=eta, pen=pen)
            except Exception as e:
                print(e)
                self.fit_batch(x, y, eta=eta, pen=pen)
            tr_eva = self.evaluate(x, y, pen=pen)
            print('times:', time)
            print('eva', tr_eva)
            te_eva = self.evaluate(x_t, y_t, pen=pen)
            tr_acc[time], tr_error[time] = tr_eva  # 记录每次学习的准确度和误差
            te_acc[time], te_error[time] = te_eva
            if acc_quit is None:
                pass
            elif eva[0] > acc_quit:
                print(f'Correctly! Acc over {acc_quit}%, quit fitting.')
                break
            elif converge_gap is not None and lasterror - eva[1] <= converge_gap:
                print(f'Converged! by acc {eva[0]}')
                break
            elif diverge_gap is not None and time > 10 and eva[1] - lasterror >= diverge_gap:
                print(f'Diverged! by acc {eva[0]}')
                break
            lasterror = eva[1]
        # 绘制学习的准确度和误差曲线
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('accuracy')
        plt.plot(np.arange(times), tr_acc, 'b', label='train acc')
        plt.plot(np.arange(times), te_acc, 'r', label='test acc')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('error')
        plt.plot(np.arange(times), tr_error, 'b', label='train error')
        plt.plot(np.arange(times), te_error, 'r', label='test error')
        plt.legend()
        plt.show()
        return tr_acc, tr_error, te_acc, te_error


# 加载mnist数据集
(tran_images, tran_labels), (test_images, test_labels) = mn.load_data()

# 学习取1000个 测试取200个
tran_images = tran_images[:1000]
tran_images = tran_images / 255
tran_labels = tran_labels[:1000]

test_images = test_images[:200]
test_images = test_images / 255
test_labels = test_labels[:200]

# 网络输入为列向量，将图像矩阵平铺成向量
length = 28 * 28
def flattenImage(image):
    return image.reshape((length,))

def reformatDataSet(dataSet):
    newSet = np.ndarray((length, dataSet.shape[0]))
    for i in range(dataSet.shape[0]):
        newSet[:, i] = flattenImage(dataSet[i])
    return newSet

tran_images_ = reformatDataSet(tran_images)
test_images_ = reformatDataSet(test_images)

# 网络输出为列向量，将图像类别标签转化成向量，只有一个元素为1
def reformatLabel(labels):
    newLabels = np.ndarray((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        newLabels[:, i] = np.zeros((10,), dtype=int)
        newLabels[labels[i], i] = 1
    return newLabels


tran_labels_ = reformatLabel(tran_labels)
test_labels_ = reformatLabel(test_labels)

model = ThisNet(length, [128, 10])

# time cost : 2 time 1 minute
args = {
    'times': 10,
    'eta': 1e-1,
    'pen': 1e-4,
    'converge_gap': None,
    'batch_size': 10,
    'acc_quit': 99.9
}
# 加载训练好的文件，不存在则重新训练
try:
    weight = np.load('ep12_weight.npy', allow_pickle=True)
    model.wList = weight
except IOError as e:
    print(e)
    model.fit(tran_images_, tran_labels_, test_images_, test_labels_, **args)
    np.save('ep12_weight', model.wList)

# result = model.getResult(test_images)
#
# print(result)

# bias & variance
# 训练误差接近测试误差：underfit， 测试误差远大于训练误差： overfit
# 教程是分别观察了 不同阶数多项式拟合 的误差，选取最优的多项式， 对于此网络可行吗？
e_tr = model.evaluate(tran_images_, tran_labels_, pen=1e-4)
e_te = model.evaluate(test_images_, test_labels_, pen=1e-4)
print(f'error of train: {e_tr}\nerror of test: {e_te}')


# def look_image(data):
#     plt.figure()
#     plt.imshow(data)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()


for i in range(200):
    plt.figure()
    plt.imshow(test_images[i])
    plt.colorbar()
    plt.grid(False)
    result = np.argmax(model.getResult(test_images_[:, i, None]))
    plt.xlabel(f"prediction of image {i} is {result}")
    plt.title(f"label of image {i} is {test_labels[i]}")
    plt.show()
