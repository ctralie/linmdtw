import time
import scipy.io as sio
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_cdf(mat, times):
    cdf = np.zeros(len(times))
    for i, time in enumerate(times):
        cdf[i] = np.sum(mat <= time)/mat.size
    return cdf

def plot_err_distributions(short = True):
    from linmdtw import get_alignment_row_col_dists
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

    distfn = get_alignment_row_col_dists
    XCPUCPU = np.zeros((N, len(times)))
    XGPUCPU = np.zeros((N, len(times)))
    XChromaFastDTW = np.zeros((N, len(times)))
    XChromaMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XMFCCFastDTW = np.zeros((N, len(times)))
    XMFCCMRMSDTW = [np.zeros((N, len(times))) for i in range(len(tauexp))]
    XChromaMFCC = np.zeros((N, len(times)))

    for i in range(N):
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
    names = ["DLNC0\nFastDTW"] + ["DLNC0\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["mfcc-mod\nFastDTW"] + ["mfcc-mod\nMRMSDTW\n$10^%i$"%exp for exp in tauexp] + ["DLNC0\nvs\nmfcc-mod"]
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
    df = pd.DataFrame({"DTW Type":approxtype, "Error (sec)":cdfthresh, "Proportion within Error Tolerance":cdf})
    ax = plt.gca()
    g = sns.swarmplot(x="DTW Type", y="Proportion within Error Tolerance", hue="Error (sec)", data=df, palette=palette)
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
    

def draw_systolic_array():
    plt.figure(figsize=(5, 5))
    AW = 0.2
    N = 6
    ax = plt.gca()
    for i in range(N):
        for j in range(N):
            if i > 0:
                ax.arrow(i-0.1, j, -0.6, 0, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
            if j > 0:
                ax.arrow(i, j-0.15, 0, -0.52, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
            if i > 0 and j > 0:
                ax.arrow(i-0.08, j-0.08, -0.67, -0.67, head_width = AW, head_length = AW, fc = 'k', ec = 'k')
    for i in range(N):
        for j in range(N):
            plt.scatter(i, j, 200, c='C0', facecolors='none', zorder=10)

    c = plt.get_cmap('afmhot')
    C = c(np.array(np.round(np.linspace(0, 255, 2*N+1)), dtype=np.int32))
    C = C[:, 0:3]
    for i in range(1, N+1):
        x = np.array([i-0.5, -0.5])
        y = np.array([-0.5, i-0.5])
        plt.plot(x, y, c=C[i-1, :], linewidth=3)
        print(i-1)
        plt.plot(N-x-1, N-y-1, c=C[2*N-i, :], linewidth=3)
        print(2*N-i)
    
    #plt.axis('off')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    f = 0.8
    ax.set_facecolor((f, f, f))
    plt.savefig("LinearSystolic.svg", bbox_inches = 'tight')

def get_length_distributions():
    fac = 0.8
    plt.figure(figsize=(fac*12, fac*3))
    for k, s in enumerate(["Short", "Long"]):
        plt.subplot(1, 2, k+1)
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        hop_length = 43
        lengths = []
        for i in range(N):
            res = json.load(open("{}/{}_0.mp3_chroma_stats.json".format(foldername, i), "r"))
            M = res['M']
            N = res['N']
            lengths.append(M/(hop_length*60))
            lengths.append(N/(hop_length*60))
        sns.distplot(np.array(lengths), kde=False, bins=20, rug=True)
        plt.xlabel("Duration (Minutes)")
        plt.ylabel("Counts")
        plt.title("{} Collection".format(s))
    plt.savefig("Counts.svg", bbox_inches='tight')



def get_cell_usage_distributions():
    ratios = []
    for s in ["Short", "Long"]:
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        for f in ["chroma", "mfcc"]:
            for i in range(N):
                res = json.load(open("{}/{}_0.mp3_{}_stats.json".format(foldername, i, f), "r"))
                denom = res['M']*res['N']
                total = res['totalCells']
                ratios.append(total/denom)
    plt.figure(figsize=(5, 3))
    sns.distplot(ratios, kde=False)
    plt.title("Ratio of cells processed to total cells")
    plt.xlabel("Ratio")
    plt.ylabel("Counts")
    plt.savefig("Cell.svg", bbox_inches='tight')


def get_memory_table():
    fac = 0.8
    delta = 30
    plt.figure(figsize=(fac*12, fac*3))
    for k, s in enumerate(["Short", "Long"]):
        plt.subplot(1, 2, k+1)
        foldername = "OrchestralPieces/{}".format(s)
        infofile = "{}/info.json".format(foldername)
        pieces = json.load(open(infofile, "r"))
        N = len(pieces)
        hop_length = 43
        lengths = []
        for i in range(N):
            res = json.load(open("{}/{}_0.mp3_chroma_stats.json".format(foldername, i), "r"))
            M = res['M']
            N = res['N']
            print(M/hop_length, pieces[i][0]['info'])
            print(N/hop_length, pieces[i][1]['info'])

            dtw = M*N*4/(1024**2)
            if dtw < 1024:
                print("DTW: ", dtw, "MB")
            else:
                print("DTW: ", dtw/1024, "GB")
            print("Ours: ", min(M, N)*4*3/(1024**2), "MB")
            print("FastDTW: ", 4*min(M, N)*(4*delta+5)/(1024**2), "MB" )

plot_err_distributions(short=True)
plot_err_distributions(short=False)
#draw_systolic_array()
#get_length_distributions()
#get_cell_usage_distributions()
#get_memory_table()