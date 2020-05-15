import time
import scipy.io as sio
import numpy as np
import json
import pandas as pd
import seaborn as sns
from AlignmentTools import *

def get_cdf(mat, times):
    cdf = np.zeros(len(times))
    for i, time in enumerate(times):
        cdf[i] = np.sum(mat <= time)/mat.size
    return cdf

def plot_err_distributions(short = True):
    foldername = "OrchestralPieces/Short"
    if not short:
        foldername = "OrchestralPieces/Long"
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))

    N = len(pieces)
    hop_size = 43
    times = np.array([1, 2, 22, 43])
    #tauexp = [3, 4, 5, 6, 7]
    tauexp = [5, 7]

    distfn = getAlignmentRowColDists
    XCPUCPU = np.zeros((N, len(times)))
    XGPUCPU = np.zeros((N, len(times)))
    XChromaFastDTW = np.zeros((N, len(times)))
    XChromaMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XMFCCFastDTW = np.zeros((N, len(times)))
    XMFCCMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XChromaMFCC = np.zeros((N, len(times)))

    for i in range(N):
        print(i, end=' ')
        res = sio.loadmat("{}/{}_0.mp3_chroma_path.mat".format(foldername, i))
        chroma_path_gpu = res['path_gpu']
        if short:
            chroma_path_cpu = res['path_cpu']

        if short:
            chroma_path_cpu64_diag = sio.loadmat("{}/{}_0.mp3_chroma_cpudiag_path.mat".format(foldername, i))['path_cpu']
            chroma_path_cpu64_left = sio.loadmat("{}/{}_0.mp3_chroma_path.mat".format(foldername, i))['path_cpu']
            d = distfn(chroma_path_cpu64_diag, chroma_path_cpu64_left)
            mfcc_path_cpu64_diag = sio.loadmat("{}/{}_0.mp3_mfcc_cpudiag_path.mat".format(foldername, i))['path_cpu']
            mfcc_path_cpu64_left = sio.loadmat("{}/{}_0.mp3_mfcc_path.mat".format(foldername, i))['path_cpu']
            d = np.concatenate((d, distfn(mfcc_path_cpu64_diag, mfcc_path_cpu64_left)))
            XCPUCPU[i, :] = get_cdf(d, times)


        res = sio.loadmat("{}/{}_0.mp3_mfcc_path.mat".format(foldername, i))
        mfcc_path_gpu = res['path_gpu']
        if short:
            mfcc_path_cpu = res['path_cpu']
        
        if short:
            d = np.concatenate((distfn(chroma_path_gpu, chroma_path_cpu), 
                            distfn(mfcc_path_gpu, mfcc_path_cpu)))
            XGPUCPU[i, :] = get_cdf(d, times)

        # Load in approximate
        res = sio.loadmat("{}/{}_0.mp3_chroma_approx_path.mat".format(foldername, i))
        XChromaFastDTW[i, :] = get_cdf(distfn(chroma_path_gpu, res['path_fastdtw']), times)
        for k, exp in enumerate(tauexp):
            XChromaMRMSDTW[k][i, :] = get_cdf(distfn(chroma_path_gpu, res['path_mrmsdtw%i'%exp]),  times)

        res = sio.loadmat("{}/{}_0.mp3_mfcc_approx_path.mat".format(foldername, i))
        XMFCCFastDTW[i, :] = get_cdf(distfn(mfcc_path_gpu, res['path_fastdtw']), times)
        for k, exp in enumerate(tauexp):
            XMFCCMRMSDTW[k][i, :] = get_cdf(distfn(mfcc_path_gpu, res['path_mrmsdtw%i'%exp]), times)

        XChromaMFCC[i, :] = get_cdf(distfn(chroma_path_gpu, mfcc_path_gpu), times)
    
    times = times/hop_size
    names = ["Chroma\nFastDTW"] + ["MFCC\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["MFCC\nFastDTW"] + ["Chroma\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["Chroma\nvs\nMFCC"]
    results = [XChromaFastDTW] + XChromaMRMSDTW + [XMFCCFastDTW] + XMFCCMRMSDTW + [XChromaMFCC]
    if short:
        names = ["CPU vs CPU\n64-bit", "GPU vs CPU\n32-bit"] + names
        results = [XCPUCPU, XGPUCPU] + results
    p = 1
    approxtype = []
    cdfthresh = []
    cdf = []
    for name, X in zip(names, results):
        for k, t in enumerate(times):
            approxtype += [name]*X.shape[0]
            cdfthresh += ["<= %.2g"%(times[k])]*X.shape[0]
            cdf += (X[:, k]**p).tolist()
    plt.figure(figsize=(6, 4))
    palette = sns.color_palette("cubehelix", len(times))
    df = pd.DataFrame({"DTW Type":approxtype, "Error (sec)":cdfthresh, "CDF":cdf})
    ax = plt.gca()
    g = sns.swarmplot(x="DTW Type", y="CDF", hue="Error (sec)", data=df, palette=palette)
    ticks = np.linspace(0, 1, 11)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%.2g"%(t**(1.0/p)) for t in ticks])
    ax.set_xticklabels(g.get_xticklabels(), rotation=90)
    if short:
        plt.title("Alignment Errors on Shorter Pieces")
        plt.savefig("Shorter.svg", bbox_inches='tight')
    else:
        plt.title("Alignment Errors on Longer Pieces")
        plt.savefig("Longer.svg", bbox_inches='tight')
    


plot_err_distributions(short=True)
plot_err_distributions(short=False)