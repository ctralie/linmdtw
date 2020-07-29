import pytest
import numpy as np
import linmdtw

class TestLibrary:
    # Does the library install in scope? Are the objects in scope?
    def test_import(self):
        import linmdtw
        from linmdtw import linmdtw, dtw_brute, dtw_brute_backtrace, fastdtw, mrmsdtw
        assert 1


class TestDTW:
    def test_input_type_warning(self):
        np.random.seed(0)
        X = np.random.rand(10, 3)
        with pytest.warns(UserWarning, match="is not 32-bit, so creating 32-bit version") as w:
            linmdtw.linmdtw(X, X)
        with pytest.warns(UserWarning, match="is not 32-bit, so creating 32-bit version") as w:
            linmdtw.dtw_brute_backtrace(X, X)

    def test_dimension_warning(self):
        np.random.seed(0)
        X = np.random.rand(3, 10)
        with pytest.warns(UserWarning, match="has more columns than rows") as w:
            linmdtw.linmdtw(X, X)
        with pytest.warns(UserWarning, match="has more columns than rows") as w:
            linmdtw.dtw_brute_backtrace(X, X)

    def test_cpu_vs_gpu(self):
        import time
        N = 2000
        t1 = np.linspace(0, 1, N)
        t2 = t1**2
        X = np.zeros((N, 2), dtype=np.float32)
        X[:, 0] = np.cos(2*np.pi*t1)
        X[:, 1] = np.sin(4*np.pi*t1)
        Y = np.zeros_like(X)
        Y[:, 0] = np.cos(2*np.pi*t2)
        Y[:, 1] = np.sin(4*np.pi*t2)
        path1 = linmdtw.linmdtw(X, Y)
        path1 = np.array(path1)
        path2 = linmdtw.dtw_brute_backtrace(X, Y)
        path2 = np.array(path2)
        err = linmdtw.get_alignment_row_col_dists(path1, path2)
        assert(np.mean(err) < 1)
        metadata = {'totalCells':0, 'M':X.shape[0], 'N':Y.shape[0], 'timeStart':time.time()}
        path3 = linmdtw.linmdtw(X, Y, do_gpu=False, metadata=metadata)
        path3 = np.array(path3)
        err = linmdtw.get_alignment_row_col_dists(path2, path3)
        assert(np.mean(err) < 1)