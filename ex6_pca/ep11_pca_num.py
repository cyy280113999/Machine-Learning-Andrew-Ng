"""
展示手写数字集的主要成分
"""
from ep1_pca import pca
from ep1_pca import pcapred
import tensorflow.keras.datasets.mnist as mn
import numpy as np
from matplotlib import pyplot as plt

# try:
#     weight = np.load('ep12_weight.npy', allow_pickle=True)
#     model.wList = weight
#     e_tr = model.evaluate(tran_images_, tran_labels_, pen=args['pen'])
#     e_te = model.evaluate(test_images_, test_labels_, pen=args['pen'])
#     print(f'error of train: {e_tr}\nerror of test: {e_te}')
# except IOError as e:
#     print(e)
#     model.fit(tran_images_, tran_labels_, test_images_, test_labels_, **args)
#     np.save('ep12_weight', model.wList)


# 加载mnist数据集
(tran_images, tran_labels), (test_images, test_labels) = mn.load_data()

# 学习取data_len个 测试取?个
data_len = 200
tran_images = tran_images[:data_len]
tran_images = tran_images / 255

test_images = test_images[:50]
test_images = test_images / 255

feature_size = (28, 28)
feature_len = 28 * 28


def reformatDataSet(dataSet):
    newSet = np.ndarray((feature_len, dataSet.shape[0]))
    for i in range(dataSet.shape[0]):
        newSet[:, i] = dataSet[i].flatten()
    return newSet


tran_images_ = reformatDataSet(tran_images)
test_images_ = reformatDataSet(test_images)

k = 100
result = pca(tran_images_, k)
tr_pj, u_r, mu, var = result['x_pj'], result['u_r'], result['mu'], result['var']

# num of k
for j in range(1):
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(u_r[:, 50*j+i].reshape(feature_size))
        plt.xlabel(50*j+i)
        plt.colorbar()
        plt.grid(False)
    plt.title('pca of 50')
    plt.show()

# num of train
for j in range(4):
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        xn_r, im_ = pcapred(tran_images_[:, [50*j+i]], u_r, mu, var)
        plt.imshow(im_.reshape(feature_size))
        plt.xlabel(50 * j + i)
        plt.colorbar()
        plt.grid(False)
    plt.title('50 of train result')
    plt.show()

# num of test
for j in range(1):
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        xn_r, im_ = pcapred(test_images_[:, [50*j+i]], u_r, mu, var)
        plt.imshow(im_.reshape(feature_size))
        plt.xlabel(50 * j + i)
        plt.colorbar()
        plt.grid(False)
    plt.title('50 of tests result')
    plt.show()
