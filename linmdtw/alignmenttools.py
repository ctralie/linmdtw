import numpy as np
from numba import jit

def get_diag_len(box, k):
    """
    Return the number of elements in this particular diagonal
    Parameters
    ----------
    MTotal: int
        Total number of samples in X
    NTotal: int
        Total number of samples in Y
    k: int
        Index of the diagonal
    Returns
    -------
    Number of elements
    """
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    starti = k
    startj = 0
    if k >= M:
        starti = M-1
        startj = k - (M-1)
    endj = k
    endi = 0
    if k >= N:
        endj = N-1
        endi = k - (N-1)
    return endj-startj+1


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

def update_alignment_metadata(metadata = None, newcells = 0):
    """
    Add new amount of cells to the total cells processed,
    and print out a percentage point if there's been progress
    Parameters
    ----------
    newcells: int
        The number of new cells that are being processed
    metadata: dictionary
        Dictionary with 'M', 'N', 'totalCells', all ints
        and 'timeStart'
    """
    if metadata:
        if 'M' in metadata and 'N' in metadata and 'totalCells' in metadata:
            import time
            denom = metadata['M']*metadata['N']
            before = np.floor(50*metadata['totalCells']/denom)
            metadata['totalCells'] += newcells
            after = np.floor(50*metadata['totalCells']/denom)
            if after > before:
                print("Parallel Alignment {}% ".format(before), end='')
                if 'timeStart' in metadata:
                    print("Elapsed time: {}".format(time.time()-metadata['timeStart']))

def get_csm(X, Y):
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

def get_ssm(X):
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
    return get_csm(X, X)

def get_path_cost(X, Y, path):
    """
    Return the cost of a warping path that matches two Euclidean 
    point clouds
    Paramters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    P1: ndarray(K, 2)
        Warping path
    """
    x = X[path[:, 0], :]
    y = Y[path[:, 1], :]
    ds = np.sqrt(np.sum((x-y)**2, 1))
    return np.sum(ds)

def make_path_strictly_increase(path):
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

def get_alignment_area_dist(P1, P2, doPlot = False):
    """
    Compute area-based alignment error.  Assume that the 
    warping paths are on the same grid
    Parameters
    ----------
    ndarray(M, 2) pP1
        First warping path
    ndarray(N, 2) pP2
        Second warping path
    doPlot: boolean
        Whether to draw a plot showing the enclosed area
    Returns
    -------
    score: float
        The area score
    """
    import scipy.sparse as sparse
    M = np.max(P1[:, 0])
    N = np.max(P1[:, 1])
    A1 = sparse.lil_matrix((M, N))
    A2 = sparse.lil_matrix((M, N))
    for i in range(P1.shape[0]):
        [ii, jj] = [P1[i, 0], P1[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A1[ii, jj::] = 1.0
    for i in range(P2.shape[0]):
        [ii, jj] = [P2[i, 0], P2[i, 1]]
        [ii, jj] = [min(ii, M-1), min(jj, N-1)]
        A2[ii, jj::] = 1.0
    A = np.abs(A1 - A2)
    dist = np.sum(A)/(M + N)
    if doPlot:
        import matplotlib.pyplot as plt
        plt.imshow(A.toarray())
        plt.scatter(P1[:, 1], P1[:, 0], 5, 'c', edgecolor = 'none')
        plt.scatter(P2[:, 1], P2[:, 0], 5, 'r', edgecolor = 'none')
        plt.title("Dist = %g"%dist)
    return dist

def get_alignment_cell_dists(P1, P2):
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
    """
    from sklearn.neighbors import KDTree, DistanceMetric
    tree = KDTree(P1, leaf_size=10)
    dists, _ = tree.query(P2, k=1)
    return dists.flatten()


@jit(nopython=True)
def get_alignment_row_dists(P1, P2):
    """
    For each point in the first path, record the distance
    of the closest point in the same row on the second path
    Parameters
    ----------
    P1: ndarray(M, 2)
        Ground truth warping path
    P2: ndarray(N, 2)
        Test warping path
    Returns
    -------
    dists: ndarray(M)
        The distances
    """
    i = 0
    k = 0
    dists = np.zeros(P1.shape[0])
    i2 = 0
    for i1 in range(P1.shape[0]):
        # Move along P2 until it's at the same row
        while P2[i2, 0] != P1[i1, 0]:
            i2 += 1
        # Check all entries of P2 that have the same row
        mindist = abs(P2[i2, 1] - P1[i1, 1])
        k = i2+1
        while k < P2.shape[0] and P2[k, 0] == P1[i1, 0]:
            mindist = min(mindist, abs(P2[k, 1]-P1[i1, 1]))
            k += 1
        dists[i1] = mindist
    return dists

def get_alignment_row_col_dists(P1, P2):
    """
    For each point in the first path, record the distance
    of the closest point in the same row on the second path,
    and vice versa.  Then repeat this along the columns
    Parameters
    ----------
    P1: ndarray(M, 2)
        Ground truth warping path
    P2: ndarray(N, 2)
        Test warping path
    Returns
    -------
    dists: ndarray(2M+2N)
        The distances
    """
    dists11 = get_alignment_row_dists(P1, P2)
    dists12 = get_alignment_row_dists(P2, P1)
    dists21 = get_alignment_row_dists(np.fliplr(P1), np.fliplr(P2))
    dists22 = get_alignment_row_dists(np.fliplr(P2), np.fliplr(P1))
    return np.concatenate((dists11, dists12, dists21, dists22))

def get_hist(dists):
    """
    Return a histogram of distances
    """
    hist = {}
    for d in dists:
        if d in hist:
            hist[d] += 1
        else:
            hist[d] = 1
    return hist

def get_inverse_fn_equally_sampled(t, x):
    import scipy.interpolate as interp
    N = len(t)
    t2 = np.linspace(np.min(x), np.max(x), N)
    try:
        res = interp.spline(x, t, t2)
        return res
    except:
        return t

def get_warp_dictionary(N, plotPaths = False):
    t = np.linspace(0, 1, N)
    D = []
    #Polynomial
    if plotPaths:
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.title('Polynomial')
    for p in range(-4, 6):
        tp = p
        if tp < 0:
            tp = -1.0/tp
        x = t**(tp**1)
        D.append(x)
        if plotPaths:
            import matplotlib.pyplot as plt
            plt.plot(x)
    #Exponential / Logarithmic
    if plotPaths:
        import matplotlib.pyplot as plt
        plt.subplot(132)
        plt.title('Exponential / Logarithmic')
    for p in range(2, 6):
        t = np.linspace(1, p**p, N)
        x = np.log(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = get_inverse_fn_equally_sampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        #D.append(x)
        #D.append(x2)
        if plotPaths:
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    #Hyperbolic Tangent
    if plotPaths:
        import matplotlib.pyplot as plt
        plt.subplot(133)
        plt.title('Hyperbolic Tangent')
    for p in range(2, 5):
        t = np.linspace(-2, p, N)
        x = np.tanh(t)
        x = x - np.min(x)
        x = x/np.max(x)
        t = t/np.max(t)
        x2 = get_inverse_fn_equally_sampled(t, x)
        x2 = x2 - np.min(x2)
        x2 = x2/np.max(x2)
        D.append(x)
        D.append(x2)
        if plotPaths:
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    D = np.array(D)
    return D

def get_warping_path_dict(D, k, doPlot = False):
    """
    Return a warping path made up of k elements
    drawn from dictionary D
    """
    N = D.shape[0]
    dim = D.shape[1]
    idxs = np.random.permutation(N)[0:k]
    weights = np.zeros(N)
    weights[idxs] = np.random.rand(k)
    weights = weights / np.sum(weights)
    res = weights.dot(D)
    res = res - np.min(res)
    res = res/np.max(res)
    if doPlot:
        import matplotlib.pyplot as plt
        plt.plot(res)
        for idx in idxs:
            plt.plot(np.arange(dim), D[idx, :], linestyle='--')
        plt.title('Constructed Warping Path')
    return res

def get_interpolated_euclidean_timeseries(X, t):
    import scipy.interpolate as interp
    M = X.shape[0]
    d = X.shape[1]
    t0 = np.linspace(0, 1, M)
    dix = np.arange(d)
    f = interp.interp2d(dix, t0, X, kind='linear')
    Y = f(dix, t)
    return Y