import numpy as np

def relabel(E):
    """
    :param E: 输入的nxk的矩阵，n是数据点的总量，k是基聚类的数量
    :return:
    """
    (n, M) = np.shape(E)
    newE = np.zeros((n, M))
    """first clustering"""
    ucl = np.unique(E[:, 0])
    if max(E[:, 0] != len(ucl)):
        for j in range(len(ucl)):
            # 初始化第一个基聚类集合的类簇，打标签
            newE[E[:, 0] == ucl[j], 0] = j + 1
            # print(newE[E[:, 0] == ucl[j]])
    # print(newE)
    """the rest of the clustering"""
    for j in range(1, M):
        ucl = np.unique(E[:, j])
        # 进行对每个基聚类的类簇计算总和
        # print(newE[:, 0:j])
        # print(np.unique(newE[:, 0:j]))
        prevCl = len(np.unique(newE[:, 0:j]))
        for i in range(len(ucl)):
            newE[E[:, j] == ucl[i], j] = prevCl + i + 1  # 进行+1操作是因为有一个是0的标签
            # print(newE[E[:, j] == ucl[i], i])
    no_allcl = max(np.unique(newE[:, -1]))
    return newE, no_allcl


def weightCl(E):
    """
    :param E:N-by-M matrix of cluster ensemble
    :return:an weighted cluster matrix
    """
    N = E.shape[0]
    no_allcl = max(np.unique(E[:, -1]))
    pc = np.zeros((N, int(no_allcl)))
    for i in range(N):
        for j in E[i, :]:
            pc[i, int(j) - 1] = 1  # 矩阵表示数据点是否属于集群（1=y，0=n），行=数据，列=cluster
    "查找每对群集的共享数据点数量==>intersect/union"
    no_allcl = int(no_allcl)
    wcl = np.zeros((no_allcl, no_allcl))
    for i in range(no_allcl):
        for j in range(i + 1, no_allcl):
            tmp = sum((pc[:, i] + pc[:, j]) > 0)
            if tmp > 0:
                wcl[i, j] = sum((pc[:, i] + pc[:, j]) == 2) / tmp
    wcl = wcl + wcl.T
    return wcl


def cts(E, dc):
    (n, M) = np.shape(E)  # n是对象个数，M是基聚类个数
    E, no_allcl = relabel(E)
    wcl = weightCl(E)
    no_allcl = int(no_allcl)
    wCT = np.zeros((no_allcl, no_allcl))
    maxCl = []
    minCl = []
    for i in range(M):
        maxCl.append(int(max(np.unique(E[:, i]))))
        minCl.append(int(min(np.unique(E[:, i]))))
    for q in range(M):
        for i in range(minCl[q], maxCl[q]):
            Ni = wcl[i - 1, :]
            aa = []
            # print("i:", i-1)
            # print("Ni:",Ni)
            for j in range(i, maxCl[q]):
                # print("j:", j)
                Nj = wcl[j, :]
                # print("Nj:", Nj)
                for ii in range(len(Nj)):
                    aa.append(min(Ni[ii], Nj[ii]))
                # print(aa)
                wCT[i, j] = sum(aa)
                # print(sum(aa))
    lwCT = []
    for i in range(len(wCT)):
        wCT[:, i]
        lwCT.append(max(wCT[:, i].tolist()))
    if max(lwCT) > 0:
        wCT = wCT / max(lwCT)
    wCT = wCT + wCT.T
    for i in range(no_allcl):
        wCT[i, i] = 1
    S = np.zeros((n, n))
    for m in range(M):
        for i in range(0, n - 1):
            for ii in range(i + 1, n):
                if E[i, m] == E[ii, m]:
                    S[i, ii] = S[i, ii] + 1
                else:
                    S[i, ii] = S[i, ii] + wCT[int(E[i, m]) - 1, int(E[ii, m]) - 1] * dc

    S = S / M
    S = S + S.T
    for i in range(n):
        S[i, i] = 1

    return S
