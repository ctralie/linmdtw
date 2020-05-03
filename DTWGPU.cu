__global__ void DTW(float* CSM, float* x, int M, int N, int diagLen, int threadsPerBlock, float* res) {
    //Have circularly rotating system of 3 buffers
    //extern __shared__ float x[]; //Circular buffer
    int off = 0;
    int upoff = 0;

    //Other local variables
    int i;
    int i1, i2, j1, j2;
    int thisi, thisj;
    int idx = threadIdx.x + blockIdx.x*threadsPerBlock;
    float val, score;

    //Initialize all buffer elements to -1
    for (off = 0; off < 3; off++) {
        if (idx < diagLen) {
            x[idx + off*diagLen] = -1;
        }
        __syncthreads();
    }
    off = 0;

    //Process each diagonal
    for (i = 0; i < N + M - 1; i++) {
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
            if (thisi < i2 || thisj > j2) {
                x[off*diagLen + idx] = -1;
            }
            else {
                x[off*diagLen + idx] = -1;

                val = CSM[thisi*N + thisj];
                score = -1;
                //Above
                if (idx + upoff + 1 < N + M - 1 && thisi > 0) {
                    if (x[((off+1)%3)*diagLen + idx + upoff + 1] > -1) {
                        score = val + x[((off+1)%3)*diagLen + idx + upoff + 1];
                    }
                }
                if (idx + upoff >= 0 && thisj > 0) {
                    //Left
                    if (x[((off+1)%3)*diagLen + idx + upoff] > -1) {
                        if (score == -1 || x[((off+1)%3)*diagLen + idx + upoff] + val < score) {
                            score = x[((off+1)%3)*diagLen + idx + upoff] + val;
                        }
                    }
                }
                if (i1 == M-1 && j1 > 1) {
                    upoff = 1;
                }
                
                if (idx + upoff >= 0 && thisi > 0) {
                    //Diagonal
                    if (x[((off+2)%3)*diagLen + idx + upoff] > -1) {
                        if (score == -1 || x[((off+2)%3)*diagLen + idx + upoff] + val < score) {
                            score = x[((off+2)%3)*diagLen + idx + upoff] + val;
                        }
                    }
                }
                
                if (score == -1) {
                    score = val;
                }
                x[off*diagLen + idx] = score;
                if (i == N + M - 2) {
                    res[0] = score;
                }
            }
            
            off = (off + 2) % 3; //Cycle buffers
        }

        __syncthreads();
    }
}
