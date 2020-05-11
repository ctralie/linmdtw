from DTW import *
from DTWGPU import *
from AlignmentTools import *
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import scipy.io as sio

def test_ordinary_vs_diag_alignment():
    """
    Test ordinary CPU backtracing against the GPU recursive
    backtracing on a highly warped example
    """
    # Setup point clouds
    initParallelAlgorithms()
    M = 8000
    t = 2*np.pi*np.linspace(0, 1, M)**2
    X = np.zeros((M, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(9*t)
    N = 6000
    t = 2*np.pi*np.linspace(0, 1, N)
    Y = np.zeros((N, 2))
    Y[:, 0] = 1.1*np.cos(t)
    Y[:, 1] = 1.1*np.sin(2*t)
    X = X
    Y = Y

    # Do ordinary DTW as a reference
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    res = DTW_Backtrace(X, Y, debug=True)
    path = res['path']
    cost = res['cost']
    print("Cost ordinary: ", cost)

    # Do parallel DTW
    #res2 = DTWDiag_GPU(X, Y)
    res2 = DTWDiag(X, Y)
    cost = res2['cost']
    print("Cost diagonal: ", cost)

    tic = time.time()
    path2 = DTWDiag_Backtrace(X, Y)
    print("Elapsed Time: ", time.time()-tic)
    path2 = np.array(path2)
    path = np.array(path)

    if path.shape[0] == path2.shape[0]:
        print("allclose: ", np.allclose(path, path2))
    else:
        print("path 1 has ", path.shape[0], " elements")
        print("path 2 has ", path2.shape[0], " elements")
    D = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        xi = X[i, :]
        xi = xi[None, :]
        D[i, :] = np.sqrt(np.sum((xi - Y)**2, 1))
    print("Cost path ordinary: ", np.sum(D[path[:, 0], path[:, 1]]))
    print("Cost path diagonal: ", np.sum(D[path2[:, 0], path2[:, 1]]))

    hist = getAlignmentCellDists(path, path2)['hist']
    print(hist)

    plt.scatter(path[:, 0], path[:, 1])
    plt.scatter(path2[:, 0], path2[:, 1], 100, marker='x')
    plt.show()


def test_timing_synthetic(dim = 20):
    import os
    from SyntheticCurves import getTorusKnot, getRandomRigidTransformation
    initParallelAlgorithms()
    np.random.seed(0)
    primes = [2, 3, 5, 7, 11, 13]
    filepath = "trials_synthetic.json"

    # Run cython code
    n_trials = 100
    MAX_SIZE = 50000
    sizes = np.linspace(0, 1, 201)[1::]**0.5
    sizes = np.round(sizes*MAX_SIZE)
    sizes = sizes[100::]
    print(sizes)
    trials = []
    if os.path.exists(filepath):
        trials = json.load(open(filepath, "r"))
    count = 0
    for N in sizes:
        N = int(N)
        WarpDict = getWarpDictionary(N)
        t1 = np.linspace(0, 1, N)
        for t in range(n_trials):
            pq = np.random.permutation(len(primes))[0:2]
            p, q = primes[pq[0]], primes[pq[1]]
            X1 = np.zeros((N, dim))
            X1[:, 0:3] = getTorusKnot(p, q, t1)
            t2 = getWarpingPath(WarpDict, 3, False)
            Y = np.zeros((N, dim))
            Y[:, 0:3] = getTorusKnot(p, q, t2)
            R, T = getRandomRigidTransformation(dim, np.std(X1))
            if count < len(trials):
                print("Skipping trial %i of %i for %i x %i"%(t+1, n_trials, N, N))
                count += 1
                continue
            
            X = np.array(X1.dot(R) + T[None, :], dtype=np.float32)
            Y = np.array(Y.dot(R) + T[None, :], dtype=np.float32)
            D = getCSM(X, Y)
            print("Doing %i x %i trial %i of %i (%i-%i torus knot)"%(N, N, t+1, n_trials, p, q))
            trial = {'p':p, 'q':q}
            # Compute score on CPU
            tic = time.time()
            trial['score_cpu'] = DTW(X, Y)['cost']
            trial['time_cpu'] = time.time() - tic

            # Compute score on GPU
            tic = time.time()
            trial['score_gpu'] = float(DTWDiag_GPU(X, Y)['cost'])
            trial['time_gpu'] = time.time() - tic

            # Compute alignment on CPU
            tic = time.time()
            path_cpu = DTW_Backtrace(X, Y)
            trial['time_align_cpu'] = time.time()-tic
            path_cpu = np.array(path_cpu)
            trial['cost_path_cpu'] = float(np.sum(D[path_cpu[:, 0], path_cpu[:, 1]]))

            # Compute alignment on GPU
            tic = time.time()
            stats = {'totalCells':0, 'M':X.shape[0], 'N':Y.shape[0], 'startTime':time.time()}
            path_gpu = DTWDiag_Backtrace(X, Y, DTWDiag_fn=DTWDiag_GPU, stats=stats)
            MN = X.shape[0]*Y.shape[0]
            perc = 100*(stats['totalCells']-MN)/MN
            print("\n%.3g Percent Cells\n\n"%perc)
            
            trial['time_align_gpu'] = time.time()-tic
            path_gpu = np.array(path_gpu)
            trial['cost_path_gpu'] = float(np.sum(D[path_gpu[:, 0], path_gpu[:, 1]]))

            # Compute alignment discrepancy
            trial['discrep1'] = getAlignmentCellDists(path_cpu, path_gpu)['hist']
            trial['discrep2'] = getAlignmentCellDists(path_gpu, path_cpu)['hist']
            trial['totalCells'] = int(stats['totalCells'])
            trial['N'] = int(N)

            print(trial)
            trials.append(trial)
            json.dump(trials, open(filepath, "w"))
            count += 1

if __name__ == '__main__':
    #test_ordinary_vs_diag_alignment()
    test_timing_synthetic()