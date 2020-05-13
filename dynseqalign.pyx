# distutils: language = c++

import numpy
cimport numpy
cimport dynseqalign
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int debug):
    cdef int M = X.shape[0]
    cdef int N = Y.shape[0]
    cdef int d = Y.shape[1]
    cdef int[:, :] P = numpy.zeros((M, N), dtype='int32')
    cdef float[:, :,] U 
    cdef float[:, :,] L
    cdef float[:, :,] UL
    if debug == 1:
        U = numpy.zeros((M, N), dtype='float32')
        L = numpy.zeros((M, N), dtype='float32')
        UL = numpy.zeros((M, N), dtype='float32')
    else:
        U = numpy.zeros((1, 1), dtype='float32')
        L = numpy.zeros((1, 1), dtype='float32')
        UL = numpy.zeros((1, 1), dtype='float32')
    cdef float[:, :,] S = numpy.zeros((M, N), dtype='float32')
    cost = dynseqalign.c_dtw(&X[0,0], &Y[0, 0], &P[0,0], M, N, d, debug, &U[0, 0], &L[0, 0], &UL[0, 0], &S[0, 0])
    ret = {'cost':cost, 'P':P}
    if debug == 1:
        ret['U'] = U
        ret['L'] = L
        ret['UL'] = UL
        ret['S'] = S
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def DTW_Diag_Step(numpy.ndarray[float,ndim=1,mode="c"] d0 not None, numpy.ndarray[float,ndim=1,mode="c"] d1 not None, numpy.ndarray[float,ndim=1,mode="c"] d2 not None, numpy.ndarray[float,ndim=1,mode="c"] csm0 not None, numpy.ndarray[float,ndim=1,mode="c"] csm1 not None, numpy.ndarray[float,ndim=1,mode="c"] csm2 not None, numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int diagLen, numpy.ndarray[int,ndim=1,mode="c"] box not None, int reverse, int i, int debug, numpy.ndarray[float,ndim=2,mode="c"] U not None, numpy.ndarray[float,ndim=2,mode="c"] L not None, numpy.ndarray[float,ndim=2,mode="c"] UL not None, numpy.ndarray[float,ndim=2,mode="c"] S not None):
    cdef int dim = X.shape[1]
    dynseqalign.c_diag_step(&d0[0], &d1[0], &d2[0], &csm0[0], &csm1[0], &csm2[0], &X[0, 0], &Y[0, 0], dim, diagLen, &box[0], reverse, i, debug, &U[0, 0], &L[0, 0], &UL[0, 0], &S[0, 0])


@cython.boundscheck(False)
@cython.wraparound(False)
def FastDTW_DynProg_Step(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, numpy.ndarray[int,ndim=1,mode="c"] I not None, numpy.ndarray[int,ndim=1,mode="c"] J not None, numpy.ndarray[int,ndim=1,mode="c"] left not None, numpy.ndarray[int,ndim=1,mode="c"] up not None, numpy.ndarray[int,ndim=1,mode="c"] diag not None, numpy.ndarray[float,ndim=1,mode="c"] S not None, numpy.ndarray[int,ndim=1,mode="c"] P not None):
    cdef int dim = X.shape[1]
    cdef int N = left.size
    dynseqalign.c_fastdtw_dynstep(&X[0, 0], &Y[0, 0], dim, &I[0], &J[0], &left[0], &up[0], &diag[0], &S[0], &P[0], N)