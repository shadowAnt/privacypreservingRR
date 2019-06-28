import numpy as np
import random


def gd(m, n, filename):
    w = []
    for i in range(n):
        w.append(random.random() * 10)
    # print(w)
    w = np.mat(w).T
    print(w.shape)
    xArr = []
    for i in range(m):
        xi = []
        for j in range(n):
            xi.append(random.random())
        y = np.mat(xi) @ w
        y += np.random.normal(0, 1)
        xi.append(y)
        xArr.append(xi)
    # xMat = np.mat(xArr)
    np.savetxt(filename, xArr, fmt='%f', delimiter='\t')
    # print(len(xArr[0]))


def gdNormal(m, n, filename):
    w = []
    for i in range(n):
        w.append(random.random() * 10)
    w = np.mat(w).T
    # print(w)

    features = np.random.normal(scale=1, size=(m, n))
    features = np.mat(features)
    noisy = np.random.normal(scale=0.01, size=(m, 1))
    y = features @ w + noisy
    xyMat = np.hstack((features, y))
    np.savetxt(filename, xyMat, fmt='%f', delimiter='\t')


if __name__ == '__main__':
    m = 50000
    n = 20
    gdNormal(m, n, './randomDataset.txt')
    gdNormal(int(m/4), n, './randomDatasetTest.txt')

