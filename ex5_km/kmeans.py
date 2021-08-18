"""
k-means cluster
k均值聚类

创建了聚类方法
创建了动态显示
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


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
    for k_, col in zip(unique_labels, colors):
        x_k = x[labels_raw == k_]
        ax1.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="y",
                 markersize=14)

    # 创建图2 显示聚类过程
    ax2 = plt.subplot(1, 2, 2)
    # cluster num , 用户可更改
    k = 5
    center = x[np.random.randint(0, data_len, (k,))]  # np.ndarray((k,dim))
    distance = np.ndarray((data_len, k))
    # label = np.ndarray((data_len,))
    iterations_max = 10
    iter_time = 0
    unique_labels = set(range(k))
    colors = plt.cm.Spectral(np.linspace(0, 1, k))
    while iter_time < iterations_max:
        for row in range(data_len):
            for col in range(k):
                distance[row, col] = np.sum(np.square(x[row] - center[col]))
        label = np.argmin(distance, axis=1)
        for k_ in range(k):
            try:
                move = np.mean(x[label == k_, :], axis=0)
            except:
                move = x[np.random.randint(0, data_len, (1,))]
            center[k_] = move

        ax2.cla()
        ax2 = plt.subplot(1, 2, 2)
        # 显示数据
        for k_, col in zip(unique_labels, colors):
            x_k = x[label == k_]
            ax2.plot(x_k[:, 0], x_k[:, 1], 'o', markerfacecolor=col, markeredgecolor="y",
                     markersize=10, label='data')
        # 显示中心
        ax2.plot(center[:, 0], center[:, 1], 'ks', label='center')
        for k_ in range(k):
            ax2.text(center[k_, 0], center[k_, 1], f'{k_} class')
        plt.pause(0.5)
        iter_time += 1

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    cluster()
