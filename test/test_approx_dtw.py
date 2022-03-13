import pytest
import numpy as np
import linmdtw

def get_pcs(N):
    t1 = np.linspace(0, 1, N)
    t2 = t1**2
    X = np.zeros((N, 2), dtype=np.float32)
    X[:, 0] = np.cos(2*np.pi*t1)
    X[:, 1] = np.sin(4*np.pi*t1)
    Y = np.zeros_like(X)
    Y[:, 0] = np.cos(2*np.pi*t2)
    Y[:, 1] = np.sin(4*np.pi*t2)
    return X, Y

class TestDTWApprox:

    def test_fasdtw(self):
        """
        Show that the fastdtw error is inversely correlated with the search
        radius
        """
        X, Y = get_pcs(1000)
        path1 = np.array(linmdtw.dtw_brute_backtrace(X, Y)[1])
        path2 = np.array(linmdtw.fastdtw(X, Y, 50)[1])
        path3 = np.array(linmdtw.fastdtw(X, Y, 5)[1])
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
    
    def test_cdtw(self):
        """
        Test constrained dynamic time warping with Sakoe-Chiba band
        """
        np.random.seed(1)
        M = 100
        N = 150
        t1 = np.linspace(0, 1, M)
        X = np.zeros((M, 2), dtype=np.float32)
        X[:, 0] = np.cos(2*np.pi*t1)
        X[:, 1] = np.sin(8*np.pi*t1)
        ## Sample an element from a dictionary of parameterizations
        ## and use this parameterization to interpolate the original
        ## time series
        D = linmdtw.alignmenttools.get_parameterization_dict(N)
        s = linmdtw.alignmenttools.sample_parameterization_dict(D, 4)
        Y = linmdtw.alignmenttools.get_interpolated_euclidean_timeseries(X, s)

        cost10, _ = linmdtw.cdtw(X, Y, 10)
        cost10_T, _ = linmdtw.cdtw(Y, X, 10)
        assert(cost10 == cost10_T)
        cost4, _ = linmdtw.cdtw(X, Y, 4)
        cost4_T, _ = linmdtw.cdtw(Y, X, 4)
        assert(cost4 == cost4_T)
        assert(cost10 < cost4)
        assert(cost10_T < cost4_T)
        

    def test_mrmsdtw(self):
        """
        Test that the error is monotonically increasing when the memory
        is decreased in mrmsdtw
        """
        X, Y = get_pcs(1000)
        path1 = linmdtw.dtw_brute_backtrace(X, Y)[1]
        path1 = np.array(path1)
        path2 = linmdtw.mrmsdtw(X, Y, tau=10**4)[1]
        path2 = np.array(path2)
        path3 = linmdtw.mrmsdtw(X, Y, tau=10**2)[1]
        path3 = np.array(path3)
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
        assert(linmdtw.get_path_cost(X, Y, path2) < linmdtw.get_path_cost(X, Y, path3))

    def test_mrmsdtw_refine(self):
        """
        Test that error is lower after refinement
        """
        X, Y = get_pcs(1000)
        path1 = linmdtw.dtw_brute_backtrace(X, Y)[1]
        path1 = np.array(path1)
        path2 = linmdtw.mrmsdtw(X, Y, tau=10**3, refine=True)[1]
        path2 = np.array(path2)
        path3 = linmdtw.mrmsdtw(X, Y, tau=10**3, refine=False)[1]
        path3 = np.array(path3)
        err1 = linmdtw.get_alignment_row_col_dists(path1, path2)
        err2 = linmdtw.get_alignment_row_col_dists(path1, path3)
        assert(np.mean(err1) <= np.mean(err2))
        assert(linmdtw.get_path_cost(X, Y, path2) < linmdtw.get_path_cost(X, Y, path3))