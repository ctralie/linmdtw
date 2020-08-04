import linmdtw
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import warnings
import json
import subprocess
import os
import glob
import youtube_dl
from scipy.spatial.distance import euclidean

def my_hook(d):
    print(d['status'])

def download_corpus(foldername):
    """
    Download a corpus of audio from youtube 
    Parameters
    ----------
    foldername: string
        Path to a folder to which to download audio.  It must 
        contain a file "info.json," with URLs, starting, and 
        stopping times of all pieces
    """
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
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except:
                warnings.warn("Error downloading {}".format(path))
                continue
            
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
    """
    Align two pieces with various techniques and save the results to .mat
    files
    Parameters
    ----------
    filename1: string
        Path to first audio file
    filename2: string
        Path to second audio file
    sr: int
        Sample rate to use on both audio files
    hop_length: int
        Hop length between windows for features
    do_mfcc: boolean
        If true, use mfcc_mod features.  Otherwise, use DLNC0 features
    compare_cpu: boolean
        If true, compare to brute force CPU DTW
    do_stretch: boolean
        If true, stretch the audio according to the GPU warping path, and save
        to a file
    delta: int
        The radius to use for fastdtw
    do_stretch_approx: boolean
        If true, stretch the audio according to approximate warping paths from
        fastdtw and mrmsdtw
    """
    if not os.path.exists(filename1):
        warnings.warn("Skipping ", filename1)
        return
    if not os.path.exists(filename2):
        warnings.warn("Skipping ", filename2)
        return
    prefix = "mfcc"
    if not do_mfcc:
        prefix = "chroma"
    pathfilename = "{}_{}_path.mat".format(filename1, prefix)
    approx_pathfilename = "{}_{}_approx_path.mat".format(filename1, prefix)
    if os.path.exists(pathfilename) and os.path.exists(approx_pathfilename):
        print("Already computed all alignments on ", filename1, filename2)
        return
    
    x1, sr = linmdtw.load_audio(filename1, sr)
    x2, sr = linmdtw.load_audio(filename2, sr)
    if do_mfcc:
        X1 = linmdtw.get_mfcc_mod(x1, sr, hop_length)
        X2 = linmdtw.get_mfcc_mod(x2, sr, hop_length)
    else:
        X1 = linmdtw.get_mixed_DLNC0_CENS(x1, sr, hop_length)
        X2 = linmdtw.get_mixed_DLNC0_CENS(x2, sr, hop_length)

    X1 = np.ascontiguousarray(X1, dtype=np.float32)
    X2 = np.ascontiguousarray(X2, dtype=np.float32)

    if os.path.exists(pathfilename):
        print("Already computed full", prefix, "alignments on", filename1, filename2)
    else:
        tic = time.time()
        metadata = {'totalCells':0, 'M':X1.shape[0], 'N':X2.shape[0], 'timeStart':tic}
        print("Starting GPU Alignment...")
        path_gpu = linmdtw.linmdtw(X1, X2, do_gpu=True, metadata=metadata)
        metadata['time_gpu'] = time.time() - metadata['timeStart']
        print("Time GPU", metadata['time_gpu'])
        path_gpu = np.array(path_gpu)
        paths = {"path_gpu":path_gpu}
        if compare_cpu:
            tic = time.time()
            print("Doing CPU alignment...")
            path_cpu = linmdtw.dtw_brute_backtrace(X1, X2)
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
            x = linmdtw.stretch_audio(x1, x2, sr, path_gpu_arr, hop_length)
            linmdtw.audiotools.save_audio(x, sr, "{}_{}_sync".format(filename1, prefix))
            print("Finished stretching")

    # Do approximate alignments
    if os.path.exists(approx_pathfilename):
        print("Already computed approximate", prefix, "alignments for", filename1, filename2)
    else:
        print("Doing fastdtw...")
        tic = time.time()
        path_fastdtw = linmdtw.fastdtw(X1, X2, radius = delta)
        elapsed = time.time()-tic
        print("Elapsed time fastdtw", elapsed)
        path_fastdtw = np.array([[p[0], p[1]] for p in path_fastdtw])
        res = {"path_fastdtw":path_fastdtw, "elapsed_fastdtw":elapsed}
        if do_stretch_approx:
            x = linmdtw.stretch_audio(x1, x2, sr, path_fastdtw, hop_length)
            linmdtw.audiotools.save_audio(x, sr, "{}_{}_fastdtw_sync".format(filename1, prefix))
        # Now do mrmsdtw with different memory restrictions
        for tauexp in [3, 4, 5, 6, 7]:
            print("Doing mrmsdtw 10^%i"%tauexp)
            tic = time.time()
            path = linmdtw.mrmsdtw(X1, X2, 10**tauexp)
            elapsed = time.time()-tic
            print("Elapsed time mrmsdtw 10^%i: %.3g"%(tauexp, elapsed))
            res['path_mrmsdtw%i'%tauexp] = path
            res['elapsed_mrmsdtw%i'%tauexp] = elapsed
        sio.savemat(approx_pathfilename, res)



def align_corpus(foldername, compare_cpu, do_stretch):
    """
    Do all of the alignments on a particular corpus
    Parameters
    ----------
    foldername: string
        Path to a folder to which to download audio.  It must 
        contain a file "info.json," with URLs, starting, and 
        stopping times of all pieces
    compare_cpu: boolean
        If true, compare to brute force CPU DTW
    do_stretch: boolean
        If true, stretch the audio according to the GPU warping path, and save
        to a file
    """
    hop_length = 512
    sr = 22050
    infofile = "{}/info.json".format(foldername)
    pieces = json.load(open(infofile, "r"))
    for do_mfcc in [False, True]:
        for i, pair in enumerate(pieces):
            try:
                filename1 = "{}/{}_0.mp3".format(foldername, i)
                filename2 = "{}/{}_1.mp3".format(foldername, i)
                print("Running alignments on ", filename1, filename2)
                align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, compare_cpu=compare_cpu, do_stretch=do_stretch)
            except:
                print("ERROR")

if __name__ == '__main__':
    download_corpus("OrchestralPieces/Short")
    align_corpus("OrchestralPieces/Short", compare_cpu=True, do_stretch=True)
    download_corpus("OrchestralPieces/Long")
    align_corpus("OrchestralPieces/Long", compare_cpu=False, do_stretch=False)