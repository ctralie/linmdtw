import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def get_diag_indices(MTotal, NTotal, k, box = None, reverse=False):
    """
    Compute the indices on a diagonal into indices on an accumulated
    distance matrix
    Parameters
    ----------
    MTotal: int
        Total number of samples in X
    NTotal: int
        Total number of samples in Y
    k: int
        Index of the diagonal
    box: list [XStart, XEnd, YStart, YEnd]
        The coordinates of the box in which to search
    Returns
    -------
    i: ndarray(dim)
        Row indices
    j: ndarray(dim)
        Column indices
    """
    if not box:
        box = [0, MTotal-1, 0, NTotal-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    starti = k
    startj = 0
    if k > M-1:
        starti = M-1
        startj = k - (M-1)
    i = np.arange(starti, -1, -1)
    j = startj + np.arange(i.size)
    dim = np.sum(j < N) # Length of this diagonal
    i = i[0:dim]
    j = j[0:dim]
    if reverse:
        i = M-1-i
        j = N-1-j
    i += box[0]
    j += box[2]
    return i, j

def update_alignment_stats(stats, newcells):
    """
    Add new amount of cells to the total cells processed,
    and print out a percentage point if there's been progress
    Parameters
    ----------
    newcells: int
        The number of new cells that are being processed
    stats: dictionary
        Dictionary with 'M', 'N', 'totalCells', all ints
        and 'timeStart'
    """
    import time
    denom = stats['M']*stats['N']
    before = np.floor(50*stats['totalCells']/denom)
    stats['totalCells'] += newcells
    after = np.floor(50*stats['totalCells']/denom)
    if after > before:
        print("Parallel Alignment {}% , at time {} ".format(before, time.time()-stats['timeStart']))
    

def getCSMCorresp(X, Y):
    """
    Return the Euclidean distance between points in
    X and Y which are in correspondence
    Paramters
    ---------
    X: ndarray(N, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Returns
    -------
    ndarray(N):
        The distances
    """
    return np.sqrt(np.sum((X - Y)**2, 1))

def getCSML1Corresp(X, Y):
    """
    Return the L1 distance between points in
    X and Y which are in correspondence
    Paramters
    ---------
    X: ndarray(N, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Returns
    -------
    ndarray(N):
        The distances
    """
    return np.sum(np.abs(X-Y), 1)

def getCSMCosineCorresp(X, Y):
    """
    Return the cosine distance between points in
    X and Y which are in correspondence
    Paramters
    ---------
    X: ndarray(N, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Returns
    -------
    ndarray(N):
        The distances
    """
    XMag = np.sqrt(np.sum(X**2, 1))
    XMag[XMag == 0] = 1
    YMag = np.sqrt(np.sum(Y**2, 1))
    YMag[YMag == 0] = 1
    return 1 - np.sum(X*Y, 1)/(XMag*YMag)

def getSplitCSMEuclideanCosineCorresp(X, Y, lam = 0.1):
    """
    Return the Euclidean distance of the first half
    plus the cosine distance of the second half
    """
    d = int(X.shape[1]/2)
    x1 = getCSMCorresp(X[:, 0:d], Y[:, 0:d])
    x2 = lam*getCSMCosineCorresp(X[:, d::], Y[:, d::])
    return x1+x2


def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between X and Y
    Paramters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    Returns
    -------
    D: ndarray(M, N)
        The cross-similarity matrix
    """
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    C = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    Parameters
    ----------
    D: ndarray(M, N)
        A cross-similarity matrix
    Kappa: float
        If Kappa = 0, take all neighbors
        If Kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise Kappa is the number of mutual neighbors to consider
    Returns
    -------
    B: ndarray(M, N)
        A binary CSM
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()

def CSMToBinaryMutual(D, Kappa):
    """
    Take the binary AND between the nearest neighbors in one direction
    and the other
    Parameters
    ----------
    D: ndarray(M, N)
        A cross-similarity matrix
    Kappa: float
        If Kappa = 0, take all neighbors
        If Kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise Kappa is the number of mutual neighbors to consider
    Returns
    -------
    B: ndarray(M, N)
        A binary CSM
    """
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1*B2

def getSSM(X):
    """
    Return the SSM between all rows of a time-ordered Euclidean point cloud X
    Paramters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Returns
    -------
    D: ndarray(M, M)
        The self-similarity matrix
    """
    return getCSM(X, X)


###################################################
#                Warping Paths                    #
###################################################
def getInverseFnEquallySampled(t, x):
    N = len(t)
    t2 = np.linspace(np.min(x), np.max(x), N)
    try:
        res = interp.spline(x, t, t2)
        return res
    except:
        return t

def getWarpDictionary(N, plotPaths = False):
    t = np.linspace(0, 1, N)
    D = []
    #Polynomial
    if plotPaths:
        plt.subplot(131)
        plt.title('Polynomial')
        plt.hold(True)
    for p in range(-4, 6):
        tp = p
        if tp < 0:
            tp = -1.0/tp
        x = t**(tp**1)
        D.append(x)
        if plotPaths:
            plt.plot(x)
    #Exponential / Logarithmic
    if plotPaths:
        plt.subplot(132)
        plt.title('Exponential / Logarithmic')
        plt.hold(True)
    for p in range(2, 6):
        t = np.linspace(1, p**p, N)
        x = np.log(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = getInverseFnEquallySampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        #D.append(x)
        #D.append(x2)
        if plotPaths:
            plt.plot(x)
            plt.plot(x2)
    #Hyperbolic Tangent
    if plotPaths:
        plt.subplot(133)
        plt.title('Hyperbolic Tangent')
        plt.hold(True)
    for p in range(2, 5):
        t = np.linspace(-2, p, N)
        x = np.tanh(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = getInverseFnEquallySampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        D.append(x)
        D.append(x2)
        if plotPaths:
            plt.plot(x)
            plt.plot(x2)
    D = np.array(D)
    return D

def getWarpingPath(D, k, doPlot = False):
    """
    Return a warping path made up of k elements
    drawn from dictionary D
    """
    N = D.shape[0]
    dim = D.shape[1]
    ret = np.zeros(dim)
    idxs = np.random.permutation(N)[0:k]
    weights = np.zeros(N)
    weights[idxs] = np.random.rand(k)
    weights = weights / np.sum(weights)
    res = weights.dot(D)
    res = res - np.min(res)
    res = res/np.max(res)
    if doPlot:
        plt.plot(res)
        plt.hold(True)
        for idx in idxs:
            plt.plot(np.arange(dim), D[idx, :], linestyle='--')
        plt.title('Constructed Warping Path')
    return res

def getInterpolatedEuclideanTimeSeries(X, t):
    M = X.shape[0]
    d = X.shape[1]
    t0 = np.linspace(0, 1, M)
    dix = np.arange(d)
    f = interp.interp2d(dix, t0, X, kind='linear')
    Y = f(dix, t)
    return Y


def projectPath(path, M, N, direction = 0):
    """
    Project the path onto one of the axes of the CSWM so that the
    correspondence is a bijection
    :param path: An NEntries x 2 array of coordinates in a warping path
    :param M: Number of rows in the CSM
    :param N: Number of columns in the CSM
    :param direction.
        0 - Choose an index along the rows to go with every
            column index
        1 - Choose an index along the columns to go with every
            row index
    :returns retpath: An NProjectedx2 array representing the projected path
    """
    involved = np.zeros((M, N))
    involved[path[:, 0], path[:, 1]] = 1
    retpath = np.array([[0, 0]], dtype =  np.int64)
    if direction == 0:
        retpath = np.zeros((N, 2), dtype = np.int64)
        retpath[:, 1] = np.arange(N)
        #Choose an index along the rows to go with every column index
        retpath[:, 0] = np.argsort(-involved, 0)[0, :]
        #Prune to the column indices that are actually used
        #(for partial matches)
        colmin = np.min(path[:, 1])
        colmax = np.max(path[:, 1])
        retpath = retpath[(retpath[:,1]>=colmin)*(retpath[:,1]<=colmax), :]
    elif direction == 1:
        retpath = np.zeros((M, 2), dtype = np.int64)
        retpath[:, 0] = np.arange(M)
        #Choose an index along the columns to go with every row index
        retpath[:, 1] = np.argsort(-involved, 1)[:, 0]
        #Prune to the row indices that are actually used
        rowmin = np.min(path[:, 0])
        rowmax = np.max(path[:, 0])
        retpath = retpath[(retpath[:,0]>=rowmin)*(retpath[:,0]<=rowmax), :]
    return retpath

def getProjectedPathParam(path, direction = 0, strcmap = 'Spectral'):
    """
    Given a projected path, return the arrays [t1, t2]
    in [0, 1] which synchronize the first curve to the
    second curve
    :param path: Nx2 array representing projected warping path
    :param direction.
        0 - There is an index along the rows to go with every
            column index
        1 - There is an index along the columns to go with every
            row index
    """
    #Figure out bounds of path
    M = path[-1, 0] - path[0, 0] + 1
    N = path[-1, 1] - path[0, 1] + 1
    if direction == 0:
        t1 = np.linspace(0, 1, M)
        t2 = (path[:, 0] - path[0, 0])/float(M)
    elif direction == 1:
        t2 = np.linspace(0, 1, N)
        t1 = float(path[:, 1] - path[0, 1])/float(N)
    else:
        print("Unknown direction for parameterizing projected paths")
        return None

    c = plt.get_cmap(strcmap)
    C1 = c(np.array(np.round(255*t1), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255*t2), dtype=np.int32))
    C2 = C2[:, 0:3]
    return {'t1':t1, 't2':t2, 'C1':C1, 'C2':C2}

def makePathStrictlyIncrease(path):
    """
    Given a warping path, remove all rows that do not
    strictly increase from the row before
    """
    toKeep = np.ones(path.shape[0])
    i0 = 0
    for i in range(1, path.shape[0]):
        if np.abs(path[i0, 0] - path[i, 0]) >= 1 and np.abs(path[i0, 1] - path[i, 1]) >= 1:
            i0 = i
        else:
            toKeep[i] = 0
    return path[toKeep == 1, :]


def rasterizeWarpingPath(P):
    if np.sum(np.abs(P - np.round(P))) == 0:
        #No effect if already integer
        return P
    P2 = np.round(P)
    P2 = np.array(P2, dtype = np.int32)
    ret = []
    for i in range(P2.shape[0]-1):
        [i1, j1] = [P2[i, 0], P2[i, 1]]
        [i2, j2] = [P2[i+1, 0], P2[i+1, 1]]
        ret.append([i1, j1])
        for k in range(1, i2-i1):
            ret.append([i1+k, j1])
        ret.append([i2, j2])
    return np.array(ret)

def computeAlignmentError(pP1, pP2, etype = 2, doPlot = False):
    """
    Compute area-based alignment error.  Assume that the 
    warping paths are on the same grid
    :param pP1: Mx2 warping path 1
    :param pP2: Nx2 warping path 2
    :param etype: Error type.  1 (default) is area ratio.  
        2 is L1 Hausdorff distance
    :param doPlot: Whether to plot the results
    """
    P1 = rasterizeWarpingPath(pP1)
    P2 = rasterizeWarpingPath(pP2)
    score = 0
    if etype == 1:
        M = np.max(P1[:, 0])
        N = np.max(P1[:, 1])
        A1 = np.zeros((M, N))
        A2 = np.zeros((M, N))
        for i in range(P1.shape[0]):
            [ii, jj] = [P1[i, 0], P1[i, 1]]
            [ii, jj] = [min(ii, M-1), min(jj, N-1)]
            A1[ii, jj::] = 1.0
        for i in range(P2.shape[0]):
            [ii, jj] = [P2[i, 0], P2[i, 1]]
            [ii, jj] = [min(ii, M-1), min(jj, N-1)]
            A2[ii, jj::] = 1.0
        A = np.abs(A1 - A2)
        score = np.sum(A)/(float(M)*float(N))
        if doPlot:
            plt.imshow(A)
            plt.hold(True)
            plt.scatter(pP1[:, 1], pP1[:, 0], 5, 'c', edgecolor = 'none')
            plt.scatter(pP2[:, 1], pP2[:, 0], 5, 'r', edgecolor = 'none')
            plt.title("Score = %g"%score)
    else:
        C = getCSM(np.array(P1, dtype = np.float32), np.array(P2, dtype = np.float32))
        score = (np.sum(np.min(C, 0)) + np.sum(np.min(C, 1)))/float(P1.shape[0]+P2.shape[0])
        if doPlot:
            plt.scatter(P1[:, 1], P1[:, 0], 20, 'c', edgecolor = 'none')
            plt.scatter(P2[:, 1], P2[:, 0], 20, 'r', edgecolor = 'none')
            idx = np.argmin(C, 1)
            for i in range(len(idx)):
                plt.plot([P1[i, 1], P2[idx[i], 1]], [P1[i, 0], P2[idx[i], 0]], 'k')
            plt.title("Score = %g"%score)
    return score


def getAlignmentCellDists(P1, P2):
    """
    Return the L1 distances between each point on the warping path
    P2 to the closest point on the warping path P1
    Parameters
    ----------
    P1: ndarray(M, 2)
        Ground truth warping path
    P2: ndarray(N, 2)
        Test warping path
    Returns
    -------
    dists: ndarray(N)
        L1 distances of each point in the second warping path
        to their closest points in p1,
    hist: dictionary{dist: count}
        A histogram of the counts
    """
    from sklearn.neighbors import KDTree
    tree = KDTree(P1)
    _, idx = tree.query(P2, k=1)
    idx = idx.flatten()
    P1Close = P1[idx, :]
    dists = np.sum(np.abs(P2-P1Close), 1)
    hist = {}
    for dist in dists:
        dist = int(dist)
        if not dist in hist:
            hist[dist] = 0
        hist[dist] += 1
    return {'dists':dists, 'hist':hist}

if __name__ == '__main__':
    #Test out alignment errors
    N = 100
    t = np.linspace(0, 1, N)
    t2 = t**2
    P1 = np.zeros((N, 2))
    P1[:, 0] = t*N
    P1[:, 1] = t2*N
    t2 = t**2.2
    P2 = np.zeros((N, 2))
    P2[:, 0] = t*N
    P2[:, 1] = t2*N
    plt.subplot(121)
    score = computeAlignmentError(P1, P2, doPlot = True)
    plt.title("Type 1 Score = %g"%score)
    plt.subplot(122)
    score = computeAlignmentError(P1, P2, 2, doPlot = True)
    plt.title("Type 2 score = %g"%score)
    plt.show()
