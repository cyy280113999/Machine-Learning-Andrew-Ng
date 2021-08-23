"""
pca 主成分分析

方法：
    pca

更新：
    main演示了pca的整个过程
    封装成pca方法

    在pcaTest中对数据初始化，采用均匀分布，正态分布，
    得到的结果令人疑惑，应探究“主成分”的具体意义
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def pcaTest2():
    x = loadmat('ex7data1.mat')['X'].T
    result = pca(x, 1)
    xp = result['x_pj']
    plt.plot(x[0],x[1],'bo')
    plt.plot(xp[0],xp[1],'bx')
    plt.show()


def pcaTest():
    # confused result
    sample_num = 10
    data_len = sample_num ** 2
    x = y = np.linspace(-1, 1, sample_num)
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    z = np.ones(shape=x.shape)
    noise_num = 10
    noise = np.random.randint(0, data_len, noise_num)
    noise_peak = 0.5
    z[noise] += (2 * np.random.rand(noise_num) - 1) * noise_peak
    plt.ion()
    ax = plt.axes(projection='3d')
    xx = np.asarray([x,y,z])
    ax.plot(x, y, z, 'bo')
    # plt.axis('equal')
    plt.title('this is noisy data')
    # plt.pause(2)

    result = pca(np.asarray([x, y, z]), 2)
    xp, mu, egv = result['x_pj'], result['mu'], result['egv']
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(xp[0], xp[1], xp[2], 'bx')
    for i in range(3):
        drawline(mu, mu + egv[:, [i]], ax=ax,type='-r')
    plt.title('this is projected data')

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(xp[0], xp[1], xp[2], 'bx')
    for i in range(3):
        drawline(mu, mu + egv[:, [i]], ax=ax,type='-r')
    ax.plot(x,y,z,'bo')
    for i in range(data_len):
        drawline(xx[:,[i]],xp[:,[i]],ax=ax,type='--k')
    plt.title('the process of projection')
    plt.ioff()
    plt.show()

def pcapred(x,u_r,mu,var):
    x_norm = x - mu
    x_norm = x_norm / var
    xn_r = u_r.T @ x_norm
    x_pj = u_r @ xn_r * var + mu
    return xn_r, x_pj

def pca(x, k):
    """
    :param x:
    :param k:
    :return: result = {
        'xn_r':xn_r,
        'x_pj':x_pj,
        'mu':mu,
        'u':un * s.reshape((un.shape[1], 1)) * var,
        'u_r':u_r
    }
    """
    mu = np.mean(x, axis=1, keepdims=True)
    x_norm = x - mu
    var = np.std(x_norm, axis=1, keepdims=True)
    var[var==0]=1e-10
    x_norm = x_norm / var
    m = x_norm.shape[1]
    covar = x_norm @ x_norm.T / m
    un, s, v = np.linalg.svd(covar)
    u_r = un[:, 0:k]
    u_r = u_r * ((2 * (u_r[-1, :] >= 0)) - 1)
    xn_r = u_r.T @ x_norm
    x_pj = u_r @ xn_r * var + mu
    result = {
        'xn_r':xn_r, # normalized reduced data
        'x_pj':x_pj, # recovered data
        'u_r': u_r,
        'mu':mu,
        'egv':un * s.reshape((un.shape[1], 1)) * var,
        'var':var
    }
    return result


def drawline(x1, x2, ax=plt, type='b'):
    if x1.shape[0] == 2:
        ax.plot([x1[0], x2[0]], [x1[1], x2[1]],type)
    elif x1.shape[0] == 3:
        points = np.hstack([x1,x2])
        ax.plot(points[0,:],
                points[1,:],
                points[2,:],type)


def main():
    step = 2
    x = loadmat('ex7data1.mat')['X'].T
    plt.ion()
    plt.plot(x[0], x[1], 'bo')
    plt.axis('equal')
    plt.xlim([0.5, 6.5])
    plt.ylim([2, 8])
    plt.xlabel('this is our raw data.')
    plt.pause(step)

    # normalize
    mu = np.mean(x, axis=1, keepdims=True)
    x_norm = x - mu
    var = np.std(x_norm, axis=1, keepdims=True)
    x_norm = x_norm / var
    plt.clf()
    plt.plot(x_norm[0], x_norm[1], 'bo')
    plt.axis('equal')
    plt.xlabel('normalized data is :')
    plt.pause(step)

    # svd
    m = x_norm.shape[1]
    covar = x_norm @ x_norm.T / m
    u, s, v = np.linalg.svd(covar)
    drawline(mu * 0, s[0] * u[:, [0]])
    drawline(mu * 0, s[1] * u[:, [1]])
    plt.xlabel('then show eigenvector on it')
    plt.pause(step)

    k = 1
    u_r = u[:, 0:k]
    u_r = u_r * ((2 * u_r[1, :] >= 0) - 1)
    x_r = u_r.T @ x_norm
    plt.clf()
    plt.plot(x_r, np.zeros(x_r.shape), 'bx')
    plt.xlabel('reduced dimension of x')
    plt.pause(step)

    x_pj = u_r @ x_r
    plt.clf()
    plt.plot(x_norm[0], x_norm[1], 'bo')
    plt.plot(x_pj[0], x_pj[1], 'bx')
    plt.axis('equal')
    plt.xlabel('recovery of x projected')
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    pcaTest()
