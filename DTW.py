import numpy as np
import matplotlib.pyplot as plt

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    D = XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)
    D[D < 0] = 0
    D = np.sqrt(D)
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

def DTWBacktrace(D, S):
    M = D.shape[0]
    N = D.shape[1]
    SR = np.zeros_like(S)
    # Fill in last three diagonals
    starti = M-1
    startj = N-1
    for diag in range(3):
        # Pull out appropriate distances
        di = np.arange(starti, -1, -1)
        dj = startj + np.arange(di.size)
        dim = np.sum(dj < M) # Length of this diagonal
        di = di[0:dim]
        dj = dj[0:dim]
        SR[di, dj] = S[di, dj]
        if startj > 0:
            startj -= 1
        else:
            starti -= 1
    plt.imshow(SR)
    plt.show()

def DTWPar(X, Y, debugging=False):
    M = X.shape[0]
    N = Y.shape[0]
    K = min(M, N)
    # The three diagonals
    d0 = np.inf*np.ones(K)
    d1 = np.inf*np.ones(K)
    d2 = np.zeros(K)
    starti = 2
    startj = 0
    # Initialize first two diagonals and helper variables
    d0[0] = np.sqrt(np.sum((X[0, :] - Y[0, :])**2))
    d1[0] = np.sqrt(np.sum((X[1, :] - Y[0, :])**2)) + d0[0]
    d1[1] = np.sqrt(np.sum((X[0, :] - Y[1, :])**2)) + d0[0]
    
    # Store the result of each r2 in memory
    S = np.array([])
    if debugging:
        S = np.zeros((M, N))
        S[0, 0] = d0[0]
        S[1, 0] = d1[0]
        S[0, 1] = d1[1]
    
    # Loop through diagonals
    for k in range(2, M+N-1):
        d2 = np.inf*np.ones(K)
        
        # Pull out appropriate distances
        i = np.arange(starti, -1, -1)
        j = startj + np.arange(i.size)
        dim = np.sum(j < N) # Length of this diagonal
        i = i[0:dim]
        j = j[0:dim]
        
        left_cost = np.inf*np.ones(dim)
        up_cost = np.inf*np.ones(dim)
        diag_cost = np.inf*np.ones(dim)
        
        ds = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
        if startj == 0:
            left_cost[1::] = d1[0:dim-1]
            # l > 0, i > 0
            idx = np.arange(dim)[(np.arange(dim) > 0)*(i > 0)] 
            diag_cost[idx] = d0[idx-1]
            # i > 0
            idx = np.arange(dim)[i > 0]
            up_cost[idx] = d1[idx]
        elif starti == X.shape[0]-1 and startj == 1:
            left_cost = d1[0:dim]
            # i > 0
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx]
            up_cost[idx] = d1[idx+1]
        elif starti == X.shape[0]-1 and startj > 1:
            left_cost = d1[0:dim]
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx+1]
            up_cost[idx] = d1[idx+1]
        
        d2[0:dim] = np.minimum(np.minimum(left_cost, diag_cost), up_cost) + ds
        if debugging:
            S[i, j] = d2[0:dim]
        if starti < M-1:
            starti += 1
        else:
            startj += 1
        d0 = np.array(d1)
        d1 = np.array(d2)
        d2 = 0*d2
    return {'cost':d2[0], 'S':S}


N = 250
t = 2*np.pi*np.linspace(0, 1, N)**2
X = np.zeros((N, 2))
X[:, 0] = np.cos(t)
X[:, 1] = np.sin(2*t)
N = 200
t = 2*np.pi*np.linspace(0, 1, N)
Y = np.zeros((N, 2))
Y[:, 0] = 1.1*np.cos(t)
Y[:, 1] = 1.1*np.sin(2*t)

res = DTW(X, Y)
path = res['path']
S = res['S']
B = res['B']

res = DTWPar(X, Y, debugging=True)
S2 = res['S']


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
