"""
k-means cluster
k均值聚类
model：
    kMeans
测试：
    cluster聚类测试
更新：
    添加了独立的kMeans方法，配合回调函数画图
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def plotCallback(x, k, label, center,
                 iter_time, error,
                 ax, pause_time):
    unique_labels = set(range(k))
    colors = plt.cm.Spectral(np.linspace(0, 1, k))
    ax.cla()
    ax.set_title(f'time:{iter_time}\ne: {error:.3f}')
    # 显示数据
    for k_, col in zip(unique_labels, colors):
        x_k = x[label == k_]
        ax.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="y",
                markersize=10, label='data')
    # 显示中心
    ax.plot(center[:, 0], center[:, 1], 'ks', label='center')
    for k_ in range(k):
        ax.text(center[k_, 0], center[k_, 1], f'{k_} class')
    plt.pause(pause_time)


def kMeans(x, k, iterations_max=10, callback=None, **args):
    """
    :param x: is matrix of (data_len , feature_num)
    :param k: num of cluster to find
    :param iterations_max:
    :param callback: function runs every iteration
    :return: label
            center
            converged? bool
    """
    data_len = x.shape[0]
    center = x[np.random.randint(0, data_len, (k,))]  # np.ndarray((k,dim))
    distance = np.ndarray((data_len, k))
    label = np.ndarray((data_len,))
    iter_time = 0
    last_error = 1e5
    converged = False
    while iter_time < iterations_max:
        for row in range(data_len):
            for col in range(k):
                distance[row, col] = np.sum(np.square(x[row] - center[col]))
        label = np.argmin(distance, axis=1)
        for k_ in range(k):
            try:
                move = np.mean(x[label == k_, :], axis=0)
            except:
                move = x[np.random.randint(0, data_len)]
            center[k_] = move
        error = np.mean(np.min(distance, axis=1))

        if callback is not None:
            fun_args = {
                'x': x,
                'k': k,
                'label': label,
                'center': center,
                'iter_time': iter_time,
                'error': error,
                **args
            }
            callback(**fun_args)
        if last_error - error <= 1e-5:
            converged = True
            break
        last_error = error
        iter_time += 1
    return (label,center,converged)

def cluster():
    # of shape (n_centers, n_features)
    data_center = np.asarray([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])

    dim = 2
    data_len = 1000
    x, labels_raw = make_blobs(n_samples=data_len, n_features=dim,
                               centers=data_center,
                               cluster_std=0.4)

    plt.ion()
    # 创建图1 显示原始数据
    unique_labels = set(labels_raw)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('raw input')
    for k_, col in zip(unique_labels, colors):
        x_k = x[labels_raw == k_]
        ax1.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="y",
                 markersize=14)

    # 创建图2 显示聚类过程
    ax2 = plt.subplot(1, 2, 2)
    # cluster num , 用户可更改
    k = 7
    kMeans(x, k, iterations_max=100, callback=plotCallback, ax=ax2, pause_time=0.1)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    cluster()