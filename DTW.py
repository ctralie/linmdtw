import numpy as np
import matplotlib.pyplot as plt

def getCSMFast(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    D = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D

def getCSM(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            D[i, j] = np.sqrt(np.sum((X[i, :] - Y[j, :])**2))
    return D

def DTW(X, Y):
    D = getCSM(X, Y)
    M = D.shape[0]
    N = D.shape[1]
    S = np.zeros((M, N))
    B = np.zeros((M, N), dtype=int) # For backtracing
    step = [[-1, -1], [-1, 0], [0, -1]] # For backtracing
    S[:, 0] = np.cumsum(D[:, 0])
    S[0, :] = np.cumsum(D[0, :])
    B[:, 0] = 1
    B[0, :] = 2
    for i in range(1, M):
        for j in range(1, N):
            xs = [S[i-1, j-1], S[i-1, j], S[i, j-1]]
            idx = np.argmin(xs)
            S[i, j] = xs[idx] + D[i, j]
            B[i, j] = idx
    #plt.imshow(B)
    #plt.show()
    path = [[M-1, N-1]]
    i = M-1
    j = N-1
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[B[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    return {'D':D, 'S':S, 'B':B, 'path':path}

def get_diag_indices(M, N, k):
    """
    Compute the indices on a diagonal into indices on an accumulated
    distance matrix
    Parameters
    ----------
    M: int
        Number of rows
    N: int
        Number of columns
    k: int
        Index of the diagonal
    Returns
    -------
    i: ndarray(dim)
        Row indices
    j: ndarray(dim)
        Column indices
    """
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
    return i, j
    

def DTWPar(X, Y, k_save = -1, k_stop = -1, debugging=False):
    """
    Parameters
    ----------
    X: ndarray(M, d)
        An M-dimensional Euclidean point cloud
    Y: ndarray(N, d)
        An N-dimensional Euclidean point cloud
    k_save: int
        Index of the diagonal d2 at which to save d0, d1, and d2
    k_stop: int
        Index of the diagonal d2 at which to stop computation
    debugging: boolean
        Whether to save the accumulated cost matrix
    
    Returns
    -------
    {
        'cost': float
            The optimal cost of the alignment (if computation didn't stop prematurely),
        'S': ndarray(M, N)
            The accumulated cost matrix (if debugging),
        'd0':ndarray(min(M, N)), 'd1':ndarray(min(M, N)), 'd2':ndarray(min(M, N))
            The saved rows if a save index was chosen
    }
    """
    M = X.shape[0]
    N = Y.shape[0]
    # Initialize first two diagonals and helper variables
    d0 = np.array([np.sqrt(np.sum((X[0, :] - Y[0, :])**2))])
    d1 = np.array([np.sqrt(np.sum((X[1, :] - Y[0, :])**2)), np.sqrt(np.sum((X[0, :] - Y[1, :])**2))]) + d0[0]
    
    # Store the result of each r2 in memory
    S = np.array([])
    if debugging:
        S = np.zeros((M, N))
        S[0, 0] = d0[0]
        S[1, 0] = d1[0]
        S[0, 1] = d1[1]
    
    # Loop through diagonals
    res = {}
    for k in range(2, M+N-1):
        i, j = get_diag_indices(M, N, k)
        dim = i.size
        d2 = np.inf*np.ones(dim)
        
        left_cost = np.inf*np.ones(dim)
        up_cost = np.inf*np.ones(dim)
        diag_cost = np.inf*np.ones(dim)
        
        # Pull out appropriate distances
        ds = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
        if j[0] == 0:
            left_cost[1::] = d1[0:dim-1]
            # l > 0, i > 0
            idx = np.arange(dim)[(np.arange(dim) > 0)*(i > 0)] 
            diag_cost[idx] = d0[idx-1]
            # i > 0
            idx = np.arange(dim)[i > 0]
            up_cost[idx] = d1[idx]
        elif i[0] == X.shape[0]-1 and j[0] == 1:
            left_cost = d1[0:dim]
            # i > 0
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx]
            up_cost[idx] = d1[idx+1]
        elif i[0] == X.shape[0]-1 and j[0] > 1:
            left_cost = d1[0:dim]
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx+1]
            up_cost[idx] = d1[idx+1]
        
        
        d2[0:dim] = np.minimum(np.minimum(left_cost, diag_cost), up_cost) + ds
        if debugging:
            S[i, j] = d2[0:dim]
        if k == k_save:
            res['d0'] = np.array(d0)
            res['d1'] = np.array(d1)
            res['d2'] = np.array(d2)
        if k == k_stop:
            break
        # Shift diagonals
        d0 = np.array(d1)
        d1 = np.array(d2)
    res['cost'] = d2[0]
    res['S'] = S
    return res

M = 250
t = 2*np.pi*np.linspace(0, 1, M)**2
X = np.zeros((M, 2))
X[:, 0] = np.cos(t)
X[:, 1] = np.sin(2*t)
N = 200
t = 2*np.pi*np.linspace(0, 1, N)
Y = np.zeros((N, 2))
Y[:, 0] = 1.1*np.cos(t)
Y[:, 1] = 1.1*np.sin(2*t)
X = X*1000
Y = Y*1000

res = DTW(X, Y)
path = res['path']
S2 = res['S']
cost = S2[-1, -1]
B = res['B']

K = X.shape[0]+Y.shape[0]-1
k_save = int(np.round(K/2.0))
res1 = DTWPar(X, Y, k_save=k_save, debugging=True)
#cost = res1['cost']
S = res1['S']

k_save_rev = k_save
if K%2 == 0:
    k_save_rev += 1
res2 = DTWPar(np.flipud(X), np.flipud(Y), k_save=k_save_rev, k_stop=k_save_rev)
res2['d0'], res2['d2'] = res2['d2'], res2['d0']
for d in ['d0', 'd1', 'd2']:
    res2[d] = res2[d][::-1]

plt.scatter(np.array([p[1] for p in path]), np.array([p[0] for p in path]))
i = 0
j = 0
l = 0
for ki, d in enumerate(['d0', 'd1', 'd2']):
    k = k_save - 2 + ki
    i, j = get_diag_indices(M, N, k)
    ds = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
    diagsum = res1[d]+res2[d]-ds
    l = np.argmin(diagsum)
    row, col = i[l], j[l]
    plt.scatter([col], [row], 100, 'red', 'x')
    plt.text(col, row, "%.5g"%diagsum[l])
plt.title("Optimal: %.5g"%cost)
j = j[l]
i = i[l]
plt.axis('equal')
plt.xlim([j-3, j+3])
plt.ylim([i-3, i+3])
plt.gca().invert_yaxis()
plt.show()
#plt.tight_layout()
#plt.savefig("DiagSums_%i_%i.svg"%(M, N), bbox_inches='tight')

"""
plt.subplot(131)
plt.imshow(S)
plt.colorbar()
plt.subplot(132)
plt.imshow(S2)
plt.colorbar()
plt.subplot(133)
plt.imshow(S-S2)
plt.colorbar()
plt.show()
"""


"""
plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.imshow(S)
plt.colorbar()
plt.subplot(222)
plt.imshow(S2)
plt.colorbar()
plt.subplot(223)
plt.imshow(S-S2)
plt.colorbar()
plt.subplot(224)
diff = np.abs(S - S2)
diff[diff > 0] = 1
plt.imshow(diff)
plt.colorbar()
plt.show()
"""

"""
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c='C0')
plt.plot(X[:, 0], X[:, 1], c='C0')
plt.scatter(Y[:, 0], Y[:, 1], c='C1')
plt.plot(Y[:, 0], Y[:, 1], c='C1')
for p in path:
    [i, j] = p
    plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k')
plt.show()
"""
