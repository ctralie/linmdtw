cdef extern from "DTW.cpp":
    float c_dtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL, float* S)

cdef extern from "DTW.cpp":
    void c_diag_step(float* d0, float* d1, float* d2, float* csm0, float* csm1, float* csm2, float* X, float* Y, int dim, int diagLen, int* box, int reverse, int i, int debug, float* U, float* L, float* UL, float* S)

cdef extern from "DTWApprox.cpp":
    float c_fastdtw_dynstep(float* X, float* Y, int d, int* I, int* J, int* left, int* up, int* diag, float* S, int* P, int N)