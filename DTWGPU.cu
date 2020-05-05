__global__ void DTW_Diag_Step(float* X, float* Y, int dim, float* d0, float* d1, float* d2, int M, int N, int diagLen, int i, int debug, float* U, float* L, float* UL) {
    int upoff = 0;

    //Other local variables
    int i1, i2, j1, j2;
    int thisi, thisj;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    float val, score, lastscore;

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
            // Step 1: Compute the Euclidean distance between Xi and Yj
            //val = CSM[thisi*N + thisj];
            val = 0.0;
            for (int k = 0; k < dim; k++) {
                double diff = X[thisi*dim + k] - Y[thisj*dim + k];
                val += diff*diff;
            }
            val = sqrt(val);

            // Step 2: Figure out the optimal cost
            //Above
            if (idx + upoff + 1 < N + M - 1 && thisi > 0) {
                lastscore = d1[idx + upoff + 1];
                if (lastscore > -1) {
                    score = val + lastscore;
                }
                if (debug == 1) {
                    U[thisi*N + thisj] = lastscore;
                }
            }
            if (idx + upoff >= 0 && thisj > 0) {
                //Left
                lastscore = d1[idx + upoff];
                if (lastscore > -1) {
                    if (score == -1 || lastscore + val < score) {
                        score = lastscore + val;
                    }
                }
                if (debug == 1) {
                    L[thisi*N + thisj] = lastscore;
                }
            }
            if (i1 == M-1 && j1 > 1) {
                upoff = 1;
            }
            if (idx + upoff >= 0 && thisi > 0) {
                //Diagonal
                lastscore = d0[idx + upoff];
                if (lastscore > -1) {
                    if (score == -1 || lastscore + val < score) {
                        score = lastscore + val;
                    }
                }
                if (debug == 1) {
                    UL[thisi*N + thisj] = lastscore;
                }
            }
            
            if (score == -1) {
                score = val;
            }
        }
        d2[idx] = score;
    }
}
