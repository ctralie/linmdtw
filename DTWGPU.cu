__global__ void DTW_Diag_Step(float* d0, float* d1, float* d2, float* csm0, float* csm1, int M, int N, int diagLen, int i, int debug, float* U, float* L, float* UL) {
    int upoff = 0;

    //Other local variables
    int i1, i2, j1, j2; // Endpoints of the diagonal
    int thisi, thisj; // Current indices on the diagonal
    float score, lastscore; // Optimal score and particular score for up/right/left
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    //Process each diagonal
    score = -1;
    if (idx < diagLen) {
        //Figure out the bounds of this diagonal
        i1 = i;
        j1 = 0;
        upoff = -1;
        if (i1 >= M) {
            i1 = M-1;
            j1 = i - (M-1);
            upoff = 0;
        }
        j2 = i;
        i2 = 0;
        if (j2 >= N) {
            j2 = N-1;
            i2 = i - (N-1);
        }
        //Update each batch
        thisi = i1 - idx;
        thisj = j1 + idx;
        if (thisi >= i2 && thisj <= j2) {
            //Figure out the optimal cost

            //Above
            if (idx + upoff + 1 < N + M - 1 && thisi > 0) {
                lastscore = d1[idx + upoff + 1];
                if (lastscore > -1) {
                    score = lastscore + csm1[idx + upoff + 1];
                }
                if (debug == 1) {
                    U[thisi*N + thisj] = lastscore;
                }
            }

            //Left
            if (idx + upoff >= 0 && thisj > 0) {
                lastscore = d1[idx + upoff];
                if (lastscore > -1) {
                    lastscore += csm1[idx + upoff];
                    if (score == -1 || lastscore < score) {
                        score = lastscore;
                    }
                }
                if (debug == 1) {
                    L[thisi*N + thisj] = lastscore;
                }
            }

            //Diagonal
            if (i1 == M-1 && j1 > 1) {
                upoff = 1;
            }
            if (idx + upoff >= 0 && thisi > 0) {
                lastscore = d0[idx + upoff];
                if (lastscore > -1) {
                    lastscore += csm0[idx + upoff];
                    if (score == -1 || lastscore < score) {
                        score = lastscore;
                    }
                }
                if (debug == 1) {
                    UL[thisi*N + thisj] = lastscore;
                }
            }
            
            if (score == -1) {
                score = 0;
            }
        }
        d2[idx] = score;
    }
}
