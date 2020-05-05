from DTW import *
from DTWGPU import *
from AlignmentTools import *
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import scipy.io as sio

def test_alignment_accuracy():
    from SyntheticCurves import getTorusKnot, getRandomRigidTransformation
    np.random.seed(0)

    M = 500
    N = 400
    #Ground truth path
    t1 = np.linspace(0, 1, M)
    t2 = np.linspace(0, 1, N)**2

    # Time-ordered point clouds
    dim = 20
    X = np.zeros((M, dim))
    X[:, 0:3] = getTorusKnot(5, 3, t1)
    Y = np.zeros((N, dim))
    Y[:, 0:3] = getTorusKnot(5, 3, t2)
    R, T = getRandomRigidTransformation(dim, np.std(X))
    X = X.dot(R) + T[None, :]
    Y = Y.dot(R) + T[None, :]
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    res = DTW_Backtrace(X, Y)

    t2 = t2*(M-1)
    t1 = (N-1)*np.linspace(0, 1, len(t2))
    PGT = np.zeros((len(t1), 2))
    PGT[:, 0] = t2
    PGT[:, 1] = t1
    P = np.array(res['path'])
    score = computeAlignmentError(PGT, P, doPlot = True)
    plt.show()

def test_ordinary_vs_diag_alignment():
    # Setup point clouds
    initParallelAlgorithms()
    M = 4000
    t = 2*np.pi*np.linspace(0, 1, M)**2
    X = np.zeros((M, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(9*t)
    N = 3000
    t = 2*np.pi*np.linspace(0, 1, N)
    Y = np.zeros((N, 2))
    Y[:, 0] = 1.1*np.cos(t)
    Y[:, 1] = 1.1*np.sin(2*t)
    X = X
    Y = Y

    # Do ordinary DTW as a reference
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    res = DTW_Backtrace(X, Y)
    path = res['path']
    cost = res['cost']
    print("Cost ordinary: ", cost)

    # Do parallel DTW
    #res2 = DTWDiag_GPU(X, Y)
    res2 = DTWDiag(X, Y)
    cost = res2['cost']
    print("Cost diagonal: ", cost)

    path2 = DTWDiag_Backtrace(X, Y, cost, DTWDiag_fn=DTWDiag_GPU, pathGT=path)
    
    path2 = np.array(path2)
    path = np.array(path)

    if path.shape[0] == path2.shape[0]:
        print("allclose: ", np.allclose(path, path2))
    else:
        print("path 1 has ", path.shape[0], " elements")
        print("path 2 has ", path2.shape[0], " elements")
    D = getCSM(X, Y)
    print("Cost path ordinary: ", np.sum(D[path[:, 0], path[:, 1]]))
    print("Cost path diagonal: ", np.sum(D[path2[:, 0], path2[:, 1]]))

    hist = getAlignmentCellDists(path, path2)['hist']
    print(hist)

    plt.scatter(path[:, 0], path[:, 1])
    plt.scatter(path2[:, 0], path2[:, 1], 100, marker='x')
    plt.show()


def test_timing(dim = 20):
    import timeit
    from SyntheticCurves import getTorusKnot, getRandomRigidTransformation
    initParallelAlgorithms()
    np.random.seed(0)

    # Run cython code
    trials = 10
    reps = 30
    MAX_SIZE = 20000
    sizes = np.linspace(0, 1, 101)[1::]**0.5
    sizes = np.round(sizes*MAX_SIZE)
    print(sizes)
    allsizes = []
    cytimes = []
    cudatimes = []
    for N in sizes:
        N = int(N)
        WarpDict = getWarpDictionary(N)
        t1 = np.linspace(0, 1, N)
        X1 = np.zeros((N, dim))
        X1[:, 0:3] = getTorusKnot(5, 3, t1)
        for trial in range(trials):
            global X
            global Y
            global path1
            global path2
            t2 = getWarpingPath(WarpDict, 3, False)
            Y = np.zeros((N, dim))
            Y[:, 0:3] = getTorusKnot(5, 3, t2)
            R, T = getRandomRigidTransformation(dim, np.std(X1))
            X = np.array(X1.dot(R) + T[None, :], dtype=np.float32)
            Y = np.array(Y.dot(R) + T[None, :], dtype=np.float32)
            print("Doing %i x %i trial %i"%(N, N, trial+1))
            cytime = timeit.timeit("DTW(X, Y)", number=reps, globals=globals())
            cudatime = timeit.timeit("DTWDiag_GPU(X, Y)", number=reps, globals=globals())
            cytime = cytime / reps
            cudatime = cudatime / reps
            print("avg cytime = ", cytime, "avg cudatime = ", cudatime)
            cytimes.append(cytime)
            cudatimes.append(cudatime)
            allsizes.append(N)
            json.dump({"cytimes":cytimes, "cudatimes":cudatimes, "allsizes":allsizes}, open("timings.txt", "w"))

def diag_cost_bug_test():
    initParallelAlgorithms()
    res = sio.loadmat("XY.mat")
    X, Y = res['X'], res['Y']
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)

   
    res1 = DTW_Backtrace(X, Y, debug=True)
    print("CPU cost", res1['cost'])

    
    res2 = DTWDiag_GPU(X, Y, debug=True)
    print("GPU cost", res2['cost'])

    path = res1['path']
    cost = res1['cost']
    res2['S'] += getCSM(X, Y)
    S = res1['S']

    path2 = DTWDiag_Backtrace(X, Y, cost, DTWDiag_fn=DTWDiag_GPU, pathGT=path)

    M = X.shape[0]
    N = Y.shape[0]
    K = M + N - 1
    # Do the forward computation
    k_save = int(np.ceil(K/2.0))
    for k in [k_save-2, k_save-1, k_save]:
        i, j = get_diag_indices(M, N, k)
        plt.plot(S[i, j])
    plt.show()

if __name__ == '__main__':
    diag_cost_bug_test()
    #test_ordinary_vs_diag_alignment()
    #test_alignment_accuracy()
    #test_timing()