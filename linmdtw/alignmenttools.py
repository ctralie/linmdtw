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
            perc = 1
            if 'perc' in metadata:
                perc = metadata['perc']
            if after > before and after % perc == 0:
                print("Parallel Alignment {}% ".format(after), end='')
                if 'timeStart' in metadata:
                    print("Elapsed time: %.3g"%(time.time()-metadata['timeStart']))

def get_csm(X, Y): # pragma: no cover
    """
    Return the Euclidean cross-similarity matrix between X and Y

    Parameters
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

def get_ssm(X): # pragma: no cover
    """
    Return the SSM between all rows of a time-ordered Euclidean point cloud X

    Parameters
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

    Parameters
    ---------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    P1: ndarray(K, 2)
        Warping path
    
    Returns
    -------
    cost: float
        The sum of the Euclidean distances along the warping path 
        between X and Y
    """
    x = X[path[:, 0], :]
    y = Y[path[:, 1], :]
    ds = np.sqrt(np.sum((x-y)**2, 1))
    return np.sum(ds)

def make_path_strictly_increase(path): # pragma: no cover
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

def refine_warping_path(path):
    """
    An implementation of the technique in section 4 of 
    "Refinement Strategies for Music Synchronization" by Ewert and Müller

    Parameters
    ----------
    path: ndarray(K, 2)
        A warping path
    
    Returns
    -------
    path_refined: ndarray(N >= K, 2)
        A refined set of correspondences
    """
    N = path.shape[0]
    ## Step 1: Identify all vertical and horizontal segments
    vert_horiz = []
    i = 0
    while i < N-1:
        if path[i+1, 0] == path[i, 0]:
            # Vertical line
            j = i+1
            while path[j, 0] == path[i, 0] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'vert', 'i':i, 'j':j})
                i = j
        elif path[i+1, 1] == path[i, 1]:
            # Horizontal line
            j = i+1
            while path[j, 1] == path[i, 1] and j < N-1:
                j += 1
            if j < N-1:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j-1})
                i = j-1
            else:
                vert_horiz.append({'type':'horiz', 'i':i, 'j':j})
                i = j
        else:
            i += 1
    
    ## Step 2: Compute local densities
    xidx = []
    density = []
    i = 0
    vhi = 0
    while i < N:
        inext = i+1
        if vhi < len(vert_horiz) and vert_horiz[vhi]['i'] == i: # This is a vertical or horizontal segment
            v = vert_horiz[vhi]
            n_seg = v['j']-v['i']+1
            xidxi = []
            densityi = []
            n_seg_prev = 0
            n_seg_next = 0
            if vhi > 0:
                v2 = vert_horiz[vhi-1]
                if i == v2['j']:
                    # First segment is at a corner
                    n_seg_prev = v2['j']-v2['i']+1
            if vhi < len(vert_horiz) - 1:
                v2 = vert_horiz[vhi+1]
                if v['j'] == v2['i']:
                    # Last segment is a corner
                    n_seg_next = v2['j']-v2['i']+1
            # Case 1: Vertical Segment
            if v['type'] == 'vert':
                xidxi = [path[i, 0] + k/n_seg for k in range(n_seg+1)]
                densityi = [n_seg]*(n_seg+1)
                if n_seg_prev > 0:
                    densityi[0] = n_seg/n_seg_prev
                if n_seg_next > 0:
                    densityi[-2] = n_seg/n_seg_next
                    densityi[-1] = n_seg/n_seg_next
                    inext = v['j']
                else:
                    inext = v['j']+1
            # Case 2: Horizontal Segment
            else:  
                xidxi = [path[i, 0] + k for k in range(n_seg)]
                densityi = [1/n_seg]*n_seg
                if n_seg_prev > 0:
                    xidxi = xidxi[1::]
                    densityi = densityi[1::]
                if n_seg_next > 0:
                    inext = v['j']
                else:
                    inext = v['j']+1
            xidx += xidxi
            density += densityi
            vhi += 1
        else:
            # This is a diagonal segment
            xidx += [path[i, 0], path[i, 0]+1]
            density += [1, 1]
        i = inext
    
    ## Step 3: Integrate densities
    xidx = np.array(xidx)
    density = np.array(density)
    path_refined = [[0, 0]]
    j = 0
    for i in range(1, xidx.size):
        if xidx[i] > xidx[i-1]:
            j += (xidx[i]-xidx[i-1])*density[i-1]
            path_refined.append([xidx[i], j])
    path_refined = np.array(path_refined)
    return path_refined

def get_alignment_area_dist(P1, P2, do_plot = False):
    """
    Compute area-based alignment error between two warping paths.

    Parameters
    ----------
    ndarray(M, 2): P1
        First warping path
    ndarray(N, 2): P2
        Second warping path
    do_plot: boolean
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
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.imshow(A.toarray())
        plt.scatter(P1[:, 1], P1[:, 0], 5, 'c', edgecolor = 'none')
        plt.scatter(P2[:, 1], P2[:, 0], 5, 'r', edgecolor = 'none')
        plt.title("Dist = %g"%dist)
    return dist


@jit(nopython=True)
def get_alignment_row_dists(P1, P2):
    """
    A measurement of errors between two warping paths.
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
        The errors at each point on the first warping path
    """
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
    A measurement of errors between two warping paths.
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
        The errors
    """
    dists11 = get_alignment_row_dists(P1, P2)
    dists12 = get_alignment_row_dists(P2, P1)
    dists21 = get_alignment_row_dists(np.fliplr(P1), np.fliplr(P2))
    dists22 = get_alignment_row_dists(np.fliplr(P2), np.fliplr(P1))
    return np.concatenate((dists11, dists12, dists21, dists22))


def get_interpolated_euclidean_timeseries(X, t, kind='linear'):
    """
    Resample a time series in Euclidean space using interp2d

    Parameters
    ----------
    X: ndarray(M, d)
        The Euclidean time series with n points
    t: ndarray(N)
        A re-parameterization function on the unit interval [0, 1]
    kind: string
        The kind of interpolation to do
    
    Returns
    -------
    Y: ndarray(N, d)
        The interpolated time series
    """
    import scipy.interpolate as interp
    M = X.shape[0]
    d = X.shape[1]
    t0 = np.linspace(0, 1, M)
    dix = np.arange(d)
    f = interp.interp2d(dix, t0, X, kind=kind)
    Y = f(dix, t)
    return Y

def get_inverse_fn_equally_sampled(t, x):
    """
    Compute the inverse of a 1D function and equally sample it.
    
    Parameters
    ---------
    t: ndarray(N)
        The domain samples of the original function
    x: ndarray(N)
        The range samples of the original function
    
    Returns
    -------
    y: ndarray(N)
        The inverse function samples
    """
    import scipy.interpolate as interp
    N = len(t)
    t2 = np.linspace(np.min(x), np.max(x), N)
    try:
        res = interp.spline(x, t, t2)
        return res
    except:
        return t

def get_parameterization_dict(N, do_plot = False):
    """
    Construct a dictionary of different types of parameterizations
    on the unit interval

    Parameters
    ----------
    N: int
        Number of samples on the unit interval
    do_plot: boolean
        Whether to plot all of the parameterizations
    
    Returns
    -------
    D: ndarray(N, K)
        The dictionary of parameterizations, with each warping path
        down a different column
    """
    t = np.linspace(0, 1, N)
    D = []
    #Polynomial
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.title('Polynomial')
    for p in range(-4, 6):
        tp = p
        if tp < 0:
            tp = -1.0/tp
        x = t**(tp**1)
        D.append(x)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.plot(x)
    #Exponential / Logarithmic
    if do_plot: # pragma: no cover
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
        D.append(x)
        D.append(x2)
        if do_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    #Hyperbolic Tangent
    if do_plot: # pragma: no cover
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
        if do_plot: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.plot(x)
            plt.plot(x2)
    D = np.array(D)
    return D

def sample_parameterization_dict(D, k, do_plot = False): 
    """
    Return a warping path made up of k random elements
    drawn from dictionary D

    Parameters
    ----------
    D: ndarray(N, K)
        The dictionary of warping paths, with each warping path
        down a different column
    k: int
        The number of warping paths to take in a combination
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
    if do_plot: # pragma: no cover
        import matplotlib.pyplot as plt
        plt.plot(res)
        for idx in idxs:
            plt.plot(np.arange(dim), D[idx, :], linestyle='--')
        plt.title('Constructed Warping Path')
    return res

def param_to_warppath(t, M):
    """
    Convert a parameterization function into a valid warping path
    
    Parameters
    ----------
    t: ndarray(N)
        Samples along the parameterization function on the
        unit interval
    M: int
        The number of samples in the original time series
    
    Returns
    -------
    P: ndarray(K, 2)
        A warping path that best matches the parameterization
    """
    N = len(t)
    P = np.zeros((N, 2), dtype=int)
    P[:, 0] = t*(M-1)
    P[:, 1] = np.arange(N)
    i = 0
    while i < P.shape[0]-1:
        diff = P[i+1, 0] - P[i, 0]
        if diff > 1:
            newchunk = np.zeros((diff-1, 2), dtype=int)
            newchunk[:, 1] = P[i, 1]
            newchunk[:, 0] = P[i, 0] + 1 + np.arange(diff-1)
            P = np.concatenate((P[0:i+1, :], newchunk, P[i+1::, :]), axis=0)
        i += 1
    return P