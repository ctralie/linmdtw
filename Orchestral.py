from DTW import *
from DTWApprox import *
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
import youtube_dl
from scipy.spatial.distance import euclidean

initParallelAlgorithms()

def my_hook(d):
    print(d['status'])

def download_corpus(foldername):
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))
    for i, pair in enumerate(pieces):
        for j, piece in enumerate(pair):
            path = "{}/{}_{}.mp3".format(foldername, i, j)
            if os.path.exists(path):
                print("Already downloaded ", path)
                continue
            url = piece['url']

            if os.path.exists('temp.mp3'):
                os.remove('temp.mp3')
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [my_hook],
                'outtmpl':'temp.mp3'
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            command = ["ffmpeg", "-i", "temp.mp3"]
            start = 0
            converting = False
            if 'start' in piece:
                converting = True
                start = piece['start']
                command.append("-ss")
                command.append("{}".format(start))
            if 'end' in piece:
                converting = True
                time = piece['end'] - start
                command.append("-t")
                command.append("{}".format(time))
            if converting:
                command.append(path)
                subprocess.call(command)
            else:
                subprocess.call(["mv", "temp.mp3", path])
    

def align_pieces(filename1, filename2, sr, hop_length, do_mfcc, compare_cpu, do_stretch=False, delta=30, do_stretch_approx=False):
    prefix = "mfcc"
    if not do_mfcc:
        prefix = "chroma"
    pathfilename = "{}_{}_path.mat".format(filename1, prefix)
    approx_pathfilename = "{}_{}_approx_path.mat".format(filename1, prefix)
    if os.path.exists(pathfilename) and os.path.exists(approx_pathfilename):
        print("Already computed all alignments on ", filename1, filename2)
        return

    x1, sr = load_audio(filename1, sr)
    x2, sr = load_audio(filename2, sr)
    if do_mfcc:
        X1 = get_mfcc_mod(x1, sr, hop_length)
        X2 = get_mfcc_mod(x2, sr, hop_length)
    else:
        X1 = get_mixed_DLNC0_CENS(x1, sr, hop_length)
        X2 = get_mixed_DLNC0_CENS(x2, sr, hop_length)

    X1 = np.ascontiguousarray(X1, dtype=np.float32)
    X2 = np.ascontiguousarray(X2, dtype=np.float32)

    if os.path.exists(pathfilename):
        print("Already computed full alignments on ", filename1, filename2)
    else:
        tic = time.time()
        metadata = {'totalCells':0, 'M':X1.shape[0], 'N':X2.shape[0], 'timeStart':tic}
        print("Starting GPU Alignment...")
        path_gpu = DTWDiag_Backtrace(X1, X2, DTWDiag_fn=DTWDiag_GPU, metadata=metadata)
        metadata['time_gpu'] = time.time() - metadata['timeStart']
        print("Time GPU", metadata['time_gpu'])
        path_gpu = np.array(path_gpu)
        paths = {"path_gpu":path_gpu}
        if compare_cpu:
            tic = time.time()
            print("Doing CPU alignment...")
            path_cpu = DTW_Backtrace(X1, X2)
            elapsed = time.time() - tic
            print("Time CPU", elapsed)
            metadata["time_cpu"] = elapsed
            path_cpu = np.array(path_cpu)
            paths["path_cpu"] = path_cpu

        for f in ['totalCells', 'M', 'N']:
            metadata[f] = int(metadata[f])
        for f in ['XGPU', 'YGPU', 'timeStart']:
            if f in metadata:
                del metadata[f]
        json.dump(metadata, open("{}_{}_stats.json".format(filename1, prefix), "w"))
        path_gpu_arr = path_gpu.copy()
        sio.savemat(pathfilename, paths)

        if do_stretch:
            print("Stretching...")
            stretch_audio(x1, x2, sr, path_gpu, hop_length, "{}_{}_sync".format(filename1, prefix))
            print("Finished stretching")

    # Do approximate alignments
    if os.path.exists(approx_pathfilename):
        print("Already computed approximate alignments for ", filename1, filename2)
    else:
        print("Doing fastdtw...")
        tic = time.time()
        path_fastdtw = fastdtw(X1, X2, radius = delta)
        elapsed = time.time()-tic
        print("Elapsed time fastdtw", elapsed)
        path_fastdtw = np.array([[p[0], p[1]] for p in path_fastdtw])
        sio.savemat(approx_pathfilename, {"path_fastdtw":path_fastdtw, "elapsed":elapsed})
        if do_stretch_approx:
            stretch_audio(x1, x2, sr, path_fastdtw, hop_length, "{}_{}_fastdtw_sync".format(filename1, prefix))


def align_corpus(foldername, compare_cpu, do_stretch):
    hop_length = 512
    sr = 22050
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))
    for do_mfcc in [False, True]:
        for i, pair in enumerate(pieces):
            try:
                filename1 = "{}/{}_0.mp3".format(foldername, i)
                filename2 = "{}/{}_1.mp3".format(foldername, i)
                print("Doing ", filename1, filename2)
                align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, compare_cpu=compare_cpu, do_stretch=do_stretch)
            except:
                print("ERROR")

if __name__ == '__main__':
    download_corpus("OrchestralPieces/Short")
    align_corpus("OrchestralPieces/Short", compare_cpu=True, do_stretch=True)
    download_corpus("OrchestralPieces/Long")
    align_corpus("OrchestralPieces/Long", compare_cpu=False, do_stretch=True)

if __name__ == '__main__2':
    hop_length = 512
    sr = 22050
    filename1 = "OrchestralPieces/Short/0_0.mp3"
    filename2 = "OrchestralPieces/Short/0_1.mp3"

    print("Doing ", filename1, filename2)
    align_pieces_approx(filename1, filename2, sr, hop_length, do_mfcc=False, do_stretch=True)