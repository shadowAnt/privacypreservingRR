import numpy as np
import matplotlib.pyplot as plt
import time

from agm import calibrateAnalyticGaussianMechanism
from generateDataset import gdNormal


def loadDataforAbalone(filename, p):
    '''
    load Data from filename, xArr is m*n, yArr is 1*m,
    where m denotes the num of sample, n denotes the num of feature.
    There's a probability of p to collect this data.
    :return: xArr, yArr. type is list.
    '''
    numFeature = len(open(filename).readline().split('\t')) - 1
    file = open(filename)
    xArr = []
    yArr = []
    for line in file.readlines():
        randNum = np.random.random()
        lineArr = []
        if (randNum <= p and randNum > 0):
            cutline = line.strip().split('\t')
            for i in range(numFeature):
                lineArr.append(float(cutline[i]))
            xArr.append(lineArr)
            yArr.append(float(cutline[-1]))
        elif (randNum > p and randNum <= 1):
            for i in range(numFeature):
                lineArr.append(0.)
            xArr.append(lineArr)
            yArr.append(0.)
        else:
            print('p ERROR!!')
            return
    return xArr, yArr


def dataNormalization(xArr, yArr):
    '''
    list->mat, (x-min)/(max-min)
    :return: xMat(m*n), yMat(m*1)   [0,1]
    '''
    xMat = np.mat(xArr)
    xMin = xMat.min(0)
    xMax = xMat.max(0)
    xMat = (xMat - xMin) / (xMax - xMin)
    yMat = np.mat(yArr).T
    yMin = yMat.min(0)
    yMax = yMat.max(0)
    yMat = (yMat - yMin) / (yMax - yMin)
    return xMat, yMat


def dataStandardization(xArr, yArr):
    '''
    list->mat, (x-mean)/(var)
    :return: xMat(m*n), yMat(m*1)
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMean = xMat.mean(0)
    xVar = xMat.var(0)
    yMean = yMat.mean(0)
    xMat = (xMat - xMean) / xVar
    yMat = yMat - yMean
    return xMat, yMat


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    (xTx+lamE).I xTy
    :param lam:
    :return:
    '''
    temp = xMat.T @ xMat / xMat.shape[0] + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(temp) == 0.0:
        print("cannot do inverse!")
        return
    w = temp.I @ xMat.T @ yMat / xMat.shape[0]
    return w


def ridgeRegresPPbyLaplace(xMat, yMat, epsilon=0.5, lam=0.2):
    '''
    by summing Xi.T @ Xi to calculate
    :return:
    '''
    m, n = xMat.shape
    xtx = np.mat(np.zeros((n, n), float))
    xty = np.mat(np.zeros((n, 1), float))
    for i in range(m):
        xtx += xMat[i].T @ xMat[i]
        xty += xMat[i].T @ yMat[i]
    # generate n*n matrix. laplace distrubution, Square symmetric matrix
    lam2 = n * (n + 3) / (2 * epsilon)
    # print('1 ' + repr(lam2))
    # print('laplace xtx  ')
    # print(xtx)
    rand_data = np.triu(np.random.laplace(0, lam2, n * n).reshape(n, n))
    # print('laplace rand_data  ')
    # print(rand_data)
    rand_data += rand_data.T - np.diag(rand_data.diagonal())
    xtx += rand_data
    xty += np.mat(np.random.laplace(0, lam2, n * 1).reshape(n, 1))
    xtx = xtx / m
    xty = xty / m
    xtx += np.eye(xMat.shape[1]) * lam
    if np.linalg.det(xtx) == 0.0:
        print("cannot do inverse!")
        return
    w = xtx.I @ xty
    return w


def ridgeRegresPPbyLaplaceWithTime(xMat, yMat, epsilon=0.5, lam=0.2):
    '''
    by summing Xi.T @ Xi to calculate
    :return:
    '''
    m, n = xMat.shape
    xtx = np.mat(np.zeros((n, n), float))
    xty = np.mat(np.zeros((n, 1), float))
    for i in range(m):
        xtx += xMat[i].T @ xMat[i]
        xty += xMat[i].T @ yMat[i]
    # generate n*n matrix. laplace distrubution, Square symmetric matrix
    lam2 = n * (n + 3) / (2 * epsilon)
    # print('1 ' + repr(lam2))
    # print('laplace xtx  ')
    # print(xtx)
    time_start = time.time()
    rand_data = np.triu(np.random.laplace(0, lam2, n * n).reshape(n, n))
    # print('laplace rand_data  ')
    # print(rand_data)
    rand_data += rand_data.T - np.diag(rand_data.diagonal())
    xtx += rand_data
    xty += np.mat(np.random.laplace(0, lam2, n * 1).reshape(n, 1))
    xtx = xtx / m
    xty = xty / m
    xtx += np.eye(xMat.shape[1]) * lam
    if np.linalg.det(xtx) == 0.0:
        print("cannot do inverse!")
        return
    w = xtx.I @ xty
    return w, time.time() - time_start


def ridgeRegresPPbyLaplaceWithP(xMat, yMat, p, epsilon=0.5, lam=0.2):
    '''
    :return:
    '''
    m, n = xMat.shape
    xtx = np.mat(np.zeros((n, n), float))
    xty = np.mat(np.zeros((n, 1), float))
    for i in range(m):
        xtx += xMat[i].T @ xMat[i]
        xty += xMat[i].T @ yMat[i]
    # generate n*n matrix. laplace distrubution, Square symmetric matrix
    khai = np.log((np.exp(epsilon) - 1) / p + 1)
    lam2 = n * (n + 3) / (2 * khai)
    # print('2 ' + repr(lam2))
    # print('laplaceWithp xtx  ')
    # print(xtx)
    rand_data = np.triu(np.random.laplace(0, lam2, n * n).reshape(n, n))
    # print('laplaceWithp rand_data  ')
    # print(rand_data)
    rand_data += rand_data.T - np.diag(rand_data.diagonal())
    xtx += rand_data
    xty += np.mat(np.random.laplace(0, lam2, n * 1).reshape(n, 1))
    xtx = xtx / (p * m)
    xty = xty / (p * m)
    xtx += np.eye(xMat.shape[1]) * lam
    if np.linalg.det(xtx) == 0.0:
        print("cannot do inverse!")
        return
    w = xtx.I @ xty
    return w


def getSigm(epsilon, delta, d):
    sigm = 1 / float(epsilon) * np.sqrt(np.log(1.25 / delta) * d * (d + 3))
    return sigm


def ridgeRegresPPbyGaussian(xMat, yMat, epsilon=0.5, lam=0.2):
    '''
    :return:
    '''
    m, n = xMat.shape
    xtx = np.mat(np.zeros((n, n), float))
    xty = np.mat(np.zeros((n, 1), float))
    for i in range(m):
        xtx += xMat[i].T @ xMat[i]
        xty += xMat[i].T @ yMat[i]
    # generate n*n matrix. laplace distrubution, Square symmetric matrix
    delta = 0.00000001
    GS = np.sqrt(n * (n + 3) / 2)
    sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, GS)
    # print(sigma)
    # print('Gaussian xtx  ')
    # print(xtx)
    rand_data = np.triu(np.random.normal(0, sigma, n * n).reshape(n, n))
    # print('Gaussion rand_data  ')
    # print(rand_data)
    rand_data += rand_data.T - np.diag(rand_data.diagonal())
    # print(rand_data)
    xtx += rand_data
    xty += np.mat(np.random.normal(0, sigma, n * 1).reshape(n, 1))
    xtx = xtx / (m)
    xty = xty / (m)
    xtx += np.eye(xMat.shape[1]) * lam
    if np.linalg.det(xtx) == 0.0:
        print("cannot do inverse!")
        return
    w = xtx.I @ xty
    return w


def ridgeRegresPPbyGaussianWithP(xMat, yMat, p, epsilon=0.5, lam=0.2):
    m, n = xMat.shape
    xtx = np.mat(np.zeros((n, n), float))
    xty = np.mat(np.zeros((n, 1), float))
    for i in range(m):
        xtx += xMat[i].T @ xMat[i]
        xty += xMat[i].T @ yMat[i]
    # generate n*n matrix. laplace distrubution, Square symmetric matrix
    delta = 0.00000001
    GS = np.sqrt(n * (n + 3) / 2)
    sigma = calibrateAnalyticGaussianMechanism(epsilon, delta, GS)
    # print(sigma)
    rand_data = np.triu(np.random.normal(0, sigma, n * n).reshape(n, n))
    rand_data += rand_data.T - np.diag(rand_data.diagonal())
    xtx += rand_data
    xty += np.mat(np.random.normal(0, sigma, n * 1).reshape(n, 1))
    xtx = xtx / (p * m)
    xty = xty / (p * m)
    xtx += np.eye(xMat.shape[1]) * lam
    if np.linalg.det(xtx) == 0.0:
        print("cannot do inverse!")
        return
    w = xtx.I @ xty
    return w


def findBestLam(filename):
    '''
    find relationship between lam and w
    :return:
    '''
    xArr, yArr = loadDataforAbalone(filename, 1)
    xMat, yMat = dataNormalization(xArr, yArr)
    numTest = 15
    wMat = np.zeros((numTest, xMat.shape[1]))
    for i in range(numTest):
        w = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = w.T
    return wMat


def plotw(wMat):
    '''
    wMat is 30*8, different lam (XTX+lam*I) has different w
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wMat)
    ax_title_text = ax.set_title('relationship between ($\lambda$) and w')
    ax_xlabel_text = ax.set_xlabel('log($\lambda$)')
    ax_ylabel_text = ax.set_ylabel('w')
    plt.setp(ax_title_text, size=20, weight='bold', color='black')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


def laplace_function(x, beta):
    '''
    give a x, return the probability of x
    :return:
    '''
    result = (1 / (2 * beta)) * np.e ** (-1 * (np.abs(x) / beta))
    return result


def Gaussion_function(x, sigm):
    result = np.exp(- x ** 2 / (2 * sigm ** 2)) / (np.sqrt(2 * np.pi) * sigm)
    return result


def draw_laplaceAndGaussion(epsilon, n):
    '''
    to draw Laplace distribution
    :return:
    '''
    lam2 = n * (n + 3) / (2 * epsilon)
    x = np.linspace(-500, 500, 40000)
    y2 = [laplace_function(x_, lam2) for x_ in x]
    GS = np.sqrt(n * (n + 3) / 2)
    delta = 0.00000001
    sigm = calibrateAnalyticGaussianMechanism(epsilon, delta, GS)
    y3 = [Gaussion_function(x_, sigm) for x_ in x]

    fig = plt.figure()
    # ax = fig.add_subplot(221)
    # ax.plot(x, y2, color='r', label='laplace')
    # ax.plot(x, y3, color='g', label='gaussion')
    # ax.set_title("probability density")

    bx = fig.add_subplot(131)
    mu = 0
    rand_data_1 = np.random.laplace(mu, lam2, n * n)
    count, bins, ignored = bx.hist(rand_data_1, n * 5, density=True)
    bx.set_title("Laplace $\lambda$ = " + str(np.around(lam2)))
    plt.xlim(-500, 500)

    cx = fig.add_subplot(132)
    rand_data_2 = np.random.normal(mu, sigm, n * n)
    count, bins, ignored = cx.hist(rand_data_2, n * 5, density=True)
    cx.set_title("Analytic Gaussion $\sigma_1$ = " + str(np.around(sigm)))
    plt.xlim(-500, 500)

    dx = fig.add_subplot(133)
    sigma = getSigm(epsilon, delta, n)
    rand_data_2 = np.random.normal(mu, sigma, n * n)
    count, bins, ignored = dx.hist(rand_data_2, n * 5, density=True)
    dx.set_title("Gaussion $\sigma_2$ = " + str(np.around(sigma)))
    plt.xlim(-500, 500)
    fig.suptitle('Probability Density, $\epsilon$=' + str(epsilon) + ', $\delta$=' + '$10^{-8}$' + ', $d$=' + str(n),
                 fontsize=20)
    plt.legend()
    plt.show()


def EuclideanDistance(xMat, yMat, w):
    yCal = xMat @ w
    yError = yMat - yCal
    # return np.sqrt(yError.T @ yError) / xMat.shape[0]
    return yError.T @ yError / xMat.shape[0]


def draw3w(xMat, yMat, w, wpp):
    '''
    y1 is True y, y2 is caled by common w, y3 is caled by Privacy-Preserving w.
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(211)
    x = [k for k in range(xMat.shape[0])]
    y1 = yMat
    y2 = xMat @ w
    y3 = xMat @ wpp
    ax.scatter(x, np.asarray((y1 - y1).T).tolist()[0], s=1, c='black', alpha=0.5, edgecolors='black', label='0')
    ax.scatter(x, np.asarray((y1 - y2).T).tolist()[0], s=1, c='coral', alpha=0.5, edgecolors='coral', label='w')
    ax.scatter(x, np.asarray((y1 - y3).T).tolist()[0], s=1, c='deepskyblue', alpha=0.5, label='wPP')
    ax.legend()
    ax_title_text = ax.set_title('relationship between wPP and w')
    ax_xlabel_text = ax.set_xlabel('number')
    ax_ylabel_text = ax.set_ylabel('Error')

    bx = fig.add_subplot(212)
    bx.plot(x, y1 - y1, c='black', linewidth=1, label='0')
    bx.plot(x, y1 - y2, c='coral', linewidth=1, label='w')
    bx.plot(x, y1 - y3, c='deepskyblue', linewidth=1, label='wPP')
    # bx_title_text = bx.set_title('relationship between wPP and w')
    bx_xlabel_text = bx.set_xlabel('number')
    bx_ylabel_text = bx.set_ylabel('Error')
    plt.legend()
    plt.show()


def drawPbyEuclideanDistance(filename, testfilename, againTime, step):
    xArr, yArr = loadDataforAbalone(filename, p=1)
    xMat, yMat = dataNormalization(xArr, yArr)
    xArrtest, yArrtest = loadDataforAbalone(testfilename, p=1)
    xMattest, yMattest = dataNormalization(xArrtest, yArrtest)
    wError = []
    wppError = []
    wpp2Error = []
    wpp3Error = []
    wpp4Error = []
    x = []
    tmp = 0.
    tmp3 = 0.
    for i in range(againTime):
        wpp = ridgeRegresPPbyLaplace(xMat, yMat)
        wpp3 = ridgeRegresPPbyGaussian(xMat, yMat)
        tmp += np.asarray(EuclideanDistance(xMattest, yMattest, wpp))[0][0]
        tmp3 += np.asarray(EuclideanDistance(xMattest, yMattest, wpp3))[0][0]
    tmp = tmp / float(againTime)
    tmp3 = tmp3 / float(againTime)

    for p in range(75, 91, step):
        print(p)
        p = p / 100
        x.append(p)
        w = ridgeRegres(xMat, yMat)
        tmp2 = 0.
        tmp4 = 0.
        for i in range(againTime):
            xArrp, yArrp = loadDataforAbalone(filename, p)
            xMatp, yMatp = dataNormalization(xArrp, yArrp)
            wpp2 = ridgeRegresPPbyLaplaceWithP(xMatp, yMatp, p)
            wpp4 = ridgeRegresPPbyGaussianWithP(xMatp, yMatp, p)
            tmp2 += np.asarray(EuclideanDistance(xMattest, yMattest, wpp2))[0][0]
            tmp4 += np.asarray(EuclideanDistance(xMattest, yMattest, wpp4))[0][0]
        tmp2 = tmp2 / float(againTime)
        tmp4 = tmp4 / float(againTime)
        wError.append(np.asarray(EuclideanDistance(xMattest, yMattest, w))[0][0])
        wppError.append(tmp)
        wpp2Error.append(tmp2)
        wpp3Error.append(tmp3)
        wpp4Error.append(tmp4)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, wError, c='black', linewidth=1, label='RR')
    ax.plot(x, wppError, c='pink', linewidth=1, label='RRpp')
    ax.plot(x, wpp2Error, c='red', linewidth=1, label='RRpp2')
    ax.plot(x, wpp3Error, c='green', linewidth=1, label='RRpp3')
    ax.plot(x, wpp4Error, c='blue', linewidth=1, label='RRpp4')

    # ax.scatter(x, wError, c='black', s=1, label='RR')
    # ax.scatter(x, wppError, c='pink', s=1, label='RRpp')
    # ax.scatter(x, wpp2Error, c='red', s=1, label='RRpp2')
    # ax.scatter(x, wpp3Error, c='green', s=1, label='RRpp3')
    # ax.scatter(x, wpp4Error, c='blue', s=1, label='RRpp4')
    ax_xlabel_text = ax.set_xlabel('p')
    ax_ylabel_text = ax.set_ylabel('Error')
    plt.legend()
    # plt.ylim(0., 5.)
    plt.show()


def distanceW(w, wpp):
    dif = w - wpp
    return dif.T @ dif / w.shape[0]


def drawPbyDistanceW(filename, againTime, step):
    xArr, yArr = loadDataforAbalone(filename, p=1)
    xMat, yMat = dataNormalization(xArr, yArr)
    wpp1Error = []
    wpp2Error = []
    wpp3Error = []
    wpp4Error = []
    x = []
    tmp1 = 0.
    tmp3 = 0.
    w = ridgeRegres(xMat, yMat)
    for i in range(againTime):
        wpp = ridgeRegresPPbyLaplace(xMat, yMat)
        wpp3 = ridgeRegresPPbyGaussian(xMat, yMat)
        tmp1 += np.asarray(distanceW(w, wpp))[0][0]
        tmp3 += np.asarray(distanceW(w, wpp3))[0][0]
    tmp1 = tmp1 / float(againTime)
    tmp3 = tmp3 / float(againTime)

    for p in range(60, 101, step):
        print(p)
        p = p / 100
        x.append(p)
        tmp2 = 0.
        tmp4 = 0.
        for i in range(againTime):
            xArrp, yArrp = loadDataforAbalone(filename, p)
            xMatp, yMatp = dataNormalization(xArrp, yArrp)
            wpp2 = ridgeRegresPPbyLaplaceWithP(xMatp, yMatp, p)
            wpp4 = ridgeRegresPPbyGaussianWithP(xMatp, yMatp, p)
            tmp2 += np.asarray(distanceW(w, wpp2))[0][0]
            tmp4 += np.asarray(distanceW(w, wpp4))[0][0]
        tmp2 = tmp2 / float(againTime)
        tmp4 = tmp4 / float(againTime)
        # wError.append(np.asarray(EuclideanDistance(xMat, yMat, w))[0][0])
        wpp1Error.append(tmp1)
        wpp2Error.append(tmp2)
        wpp3Error.append(tmp3)
        wpp4Error.append(tmp4)
    print(wpp1Error)
    print(wpp2Error)
    print(wpp3Error)
    print(wpp4Error)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, wpp1Error, c='pink', linewidth=1, label='RRpp')
    ax.plot(x, wpp2Error, c='red', linewidth=1, label='RRpp2')
    ax.plot(x, wpp3Error, c='green', linewidth=1, label='RRpp3')
    ax.plot(x, wpp4Error, c='blue', linewidth=1, label='RRpp4')

    # ax.scatter(x, wError, c='black', s=1, label='RR')
    # ax.scatter(x, wppError, c='pink', s=1, label='RRpp')
    # ax.scatter(x, wpp2Error, c='red', s=1, label='RRpp2')
    # ax.scatter(x, wpp3Error, c='green', s=1, label='RRpp3')
    # ax.scatter(x, wpp4Error, c='blue', s=1, label='RRpp4')
    ax_xlabel_text = ax.set_xlabel('p')
    ax_ylabel_text = ax.set_ylabel('Error')
    plt.legend()
    # plt.ylim(0., 30.)
    plt.show()

    # draw3w(xMat, yMat, w, wpp2)


def testTime(filename):
    t = []
    x = [k for k in range(8, 30)]
    for i in range(8, 30):
        print(i)
        gdNormal(100, i, filename)
        xArr, yArr = loadDataforAbalone(filename, 1)
        xMat, yMat = dataNormalization(xArr, yArr)
        # time_start = time.time()
        sum = 0
        for k in range(1000):
            wpp, time_k = ridgeRegresPPbyLaplaceWithTime(xMat, yMat)
            sum += time_k
        # time_end = time.time()
        time_cost = sum / 1000.
        t.append(time_cost)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[1:], t[1:], c='red', linewidth=1)
    ax.set_xlabel('dimension $d$', size=10, weight='bold')
    ax.set_ylabel('Time /s', size=10, weight='bold')
    ax.set_title('Relationship between dimension $d$ and Time', size=20, weight='bold')
    plt.legend()
    # plt.ylim(0., 30.)
    plt.show()


def testDandError(filename, testfilename, againTime):
    disWPPlist = []
    disWPP3list = []

    yErrorlist = []
    yErrorpplist = []
    yErrorpp3list = []
    m = 50000
    for i in range(20, 31, 2):
        print(i)
        gdNormal(m, i, filename)
        gdNormal(int(m / 4), i, testfilename)
        xArr, yArr = loadDataforAbalone(filename, 1)
        xMat, yMat = dataNormalization(xArr, yArr)
        xArrtest, yArrtest = loadDataforAbalone(testfilename, p=1)
        xMattest, yMattest = dataNormalization(xArrtest, yArrtest)
        w = ridgeRegres(xMat, yMat)
        disWPP = 0
        disWPP3 = 0
        yError = np.asarray(EuclideanDistance(xMattest, yMattest, w))[0][0]
        yErrorpp = 0
        yErrorpp3 = 0
        upper = 1500
        for j in range(againTime):
            wpp = ridgeRegresPPbyLaplace(xMat, yMat)
            wpp3 = ridgeRegresPPbyGaussian(xMat, yMat)
            disWPP += min(np.asarray(distanceW(w, wpp))[0][0], upper)
            disWPP3 += min(np.asarray(distanceW(w, wpp3))[0][0], upper)

            yErrorpp += min(np.asarray(EuclideanDistance(xMattest, yMattest, wpp))[0][0], upper)
            yErrorpp3 += min(np.asarray(EuclideanDistance(xMattest, yMattest, wpp3))[0][0], upper)
        disWPP = disWPP / float(againTime)
        disWPP3 = disWPP3 / float(againTime)

        yErrorpp = yErrorpp / float(againTime)
        yErrorpp3 = yErrorpp3 / float(againTime)
        disWPPlist.append(disWPP)
        disWPP3list.append(disWPP3)

        yErrorlist.append(yError)
        yErrorpplist.append(yErrorpp)
        yErrorpp3list.append(yErrorpp3)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    x = [k for k in range(20, 31, 2)]
    ax.plot(x, disWPPlist, c='pink', linewidth=1, label='RRppLaplace')
    print(disWPPlist)
    print()
    ax.plot(x, disWPP3list, c='red', linewidth=1, label='RRppGaussion')
    print(disWPP3list)
    print()
    ax_xlabel_text = ax.set_xlabel('Dimension', size=10, weight='bold')
    ax_ylabel_text = ax.set_ylabel('trainError', size=10, weight='bold')
    ax.set_title('Relationship between dimension $d$ and trainError', size=20, weight='bold')

    bx = fig.add_subplot(212)
    # bx.plot(x, yErrorlist, c='pink', linewidth=1, label='RR')
    bx.plot(x, yErrorpplist, c='pink', linewidth=1, label='RRppLaplace')
    print(yErrorpplist)
    print()
    bx.plot(x, yErrorpp3list, c='red', linewidth=1, label='RRppGaussion')
    print(yErrorpp3list)
    bx_xlabel_text = bx.set_xlabel('Dimension', size=10, weight='bold')
    bx_ylabel_text = bx.set_ylabel('testError', size=10, weight='bold')
    bx.set_title('Relationship between dimension $d$ and testError', size=20, weight='bold')

    plt.legend()
    # plt.ylim(0., 30.)
    plt.show()


if __name__ == '__main__':
    # TODO ridgeRegression
    # filename = './abalone.txt'
    filename = './randomDataset.txt'
    testfilename = './randomDatasetTest.txt'
    againTime = 20
    step = 1
    testDandError(filename, testfilename, againTime)
    #
    # wMat = findBestLam(filename)
    # plotw(wMat)

    # TODO laplace
    # xArr, yArr = loadDataforAbalone(filename, 1)
    # xMat, yMat = dataNormalization(xArr, yArr)
    # w = ridgeRegres(xMat, yMat)
    # wpp = ridgeRegresPPbyGaussian(xMat, yMat)
    # print(EuclideanDistance(xMat, yMat, w))
    # print(EuclideanDistance(xMat, yMat, wpp))
    # draw3w(xMat, yMat, w, wpp)

    # drawPbyDistanceW(filename, testfilename, againTime, step)
    # drawPbyEuclideanDistance(filename, testfilename, againTime, step)
    # draw_laplaceAndGaussion(epsilon=0.5, n=20)
    #
    # testTime(filename)
