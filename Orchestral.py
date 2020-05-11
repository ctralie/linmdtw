from DTW import *
from DTWGPU import *
from AlignmentTools import *
from AudioTools import *
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pyrubberband as pyrb
import subprocess
import os
import glob

initParallelAlgorithms()

def align_pieces(filename1, filename2, sr, hop_length, do_mfcc, do_cpu, do_stretch=False):
    x1, sr = load_audio(filename1, sr)
    x2, sr = load_audio(filename2, sr)
    prefix = "mfcc"
    if do_mfcc:
        X1 = get_mfcc_mod(x1, sr, hop_length)
        X2 = get_mfcc_mod(x2, sr, hop_length)
    else:
        prefix = "chroma"
        X1 = get_mixed_DLNC0_CENS(x1, sr, hop_length)
        X2 = get_mixed_DLNC0_CENS(x2, sr, hop_length)


    X1 = np.ascontiguousarray(X1, dtype=np.float32)
    X2 = np.ascontiguousarray(X2, dtype=np.float32)

    dist_fn = getCSMCorresp

    tic = time.time()
    stats = {'totalCells':0, 'M':X1.shape[0], 'N':X2.shape[0], 'timeStart':tic}
    print("Starting GPU Alignment...")
    path_gpu = DTWDiag_Backtrace(X1, X2, DTWDiag_fn=DTWDiag_GPU, stats=stats, dist_fn=dist_fn)
    print("Time GPU", time.time()-tic)
    path_gpu = np.array(path_gpu)

    if do_cpu:
        tic = time.time()
        path_cpu = DTW_Backtrace(X1, X2, dist_fn=dist_fn)
        print("Time CPU", time.time()-tic)
        path_cpu = np.array(path_cpu)
        hist1 = getAlignmentCellDists(path_cpu, path_gpu)['hist']
        hist2 = getAlignmentCellDists(path_gpu, path_cpu)['hist']
        json.dump([hist1, hist2], open("%s_gpucpu.json", "w"))

    path_gpu_arr = path_gpu.copy()
    sio.savemat("{}_{}path.mat".format(filename1, prefix), {"path_gpu":path_gpu})

    path_gpu *= hop_length
    path_gpu = [(row[0], row[1]) for row in path_gpu]
    path_gpu.append((x1.size, x2.size))

    if do_stretch:
        print("Stretching...")
        x1_stretch = pyrb.timemap_stretch(x1, sr, path_gpu)
        x3 = np.zeros((x2.size, 2))
        x3[:, 0] = x2
        x3[:, 1] = x1_stretch 
        wavfilename = "{}_{}sync.wav".format(filename1, prefix)
        mp3filename = "{}_{}sync.mp3".format(filename1, prefix)
        if os.path.exists(wavfilename):
            os.remove(wavfilename)
        if os.path.exists(mp3filename):
            os.remove(mp3filename)
        wavfile.write(wavfilename, sr, x3)
        subprocess.call(["ffmpeg", "-i", wavfilename, mp3filename])
        os.remove(wavfilename)

hop_length = 512
sr = 22050
for do_mfcc in [False, True]:
    for i in range(1, 8):
        filename1 = "OrchestralPieces/Short/%i_1.webm"%i
        filename2 = "OrchestralPieces/Short/%i_2.webm"%i
        print("Doing ", filename1, filename2)
        align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, do_cpu=True, do_stretch=True)
    for i in range(1, 5):
        filename1 = glob.glob("OrchestralPieces/Long/%i_1*"%i)[0]
        filename2 = glob.glob("OrchestralPieces/Long/%i_2*"%i)[0]
        align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, do_cpu=False, do_stretch=False)