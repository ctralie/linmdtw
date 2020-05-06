__global__ void DTW_Diag_Step(float* d0, float* d1, float* d2, float* csm0, float* csm1, int M, int N, int diagLen, int i, int debug, float* U, float* L, float* UL, float* S) {
    int upoff = 0;

    //Other local variables
    int i1, i2, j1, j2; // Endpoints of the diagonal
    int thisi, thisj; // Current indices on the diagonal
    // Optimal score and particular score for up/right/left
    float score, left, up, diag; 
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    //Process each diagonal
    score = -1;
    if (idx < diagLen) {
        //Figure out the bounds of this diagonal
        i1 = i;
        j1 = 0;
        if (i1 >= M) {
            i1 = M-1;
            j1 = i - (M-1);
        }
        j2 = i;
        i2 = 0;
        if (j2 >= N) {
            j2 = N-1;
            i2 = i - (N-1);
        }
        //Update each batch  (1, 0), (0, 1)
        thisi = i1 - idx;
        thisj = j1 + idx;
        if (thisi >= i2 && thisj <= j2) {
            //Figure out the optimal cost
            if (thisi == 0 && thisj == 0) {
                score = 0;
                if (debug == -1) {
                    S[0] = 0;
                    U[0] = -1;
                    L[0] = -1;
                    UL[0] = -1;
                }
            }
            else {
                left = -1;
                up = -1;
                diag = -1;
                if (j1 == 0) {
                    if (idx > 0) {
                        left = d1[idx-1] + csm1[idx-1];
                    }
                    if (idx > 0 && thisi > 0) {
                        diag = d0[idx-1] + csm0[idx-1];
                    }
                    if (thisi > 0) {
                        up = d1[idx] + csm1[idx];
                    }
                }
                else if (i1 == M-1 && j1 == 1) {
                    left = d1[idx] + csm1[idx];
                    if (thisi > 0) {
                        diag = d0[idx] + csm0[idx];
                        up = d1[idx+1] + csm1[idx+1];
                    }
                }
                else if (i1 == M-1 && j1 > 1) {
                    left = d1[idx] + csm1[idx];
                    if (thisi > 0) {
                        diag = d0[idx+1] + csm0[idx+1];
                        up = d1[idx+1] + csm1[idx+1];
                    }
                }
                if (left > -1) {
                    score = left;
                }
                if (up > -1 && (up < score || score == -1)) {
                    score = up;
                }
                if (diag > -1 && (diag < score || score == -1)) {
                    score = diag;
                }
                if (debug == 1) {
                    U[thisi*N + thisj] = up;
                    L[thisi*N + thisj] = left;
                    UL[thisi*N + thisj] = diag;
                    S[thisi*N + thisj] = score;
                }
            }
        }
        d2[idx] = score;
    }
}
