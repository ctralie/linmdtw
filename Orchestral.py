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
import youtube_dl

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
                print("Skipping ", path)
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
    

def align_pieces(filename1, filename2, sr, hop_length, do_mfcc, do_cpu, do_stretch=False):
    prefix = "mfcc"
    if not do_mfcc:
        prefix = "chroma"
    pathfilename = "{}_{}_path.mat".format(filename1, prefix)
    if os.path.exists(pathfilename):
        print("Skipping", filename1, filename2)
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

    tic = time.time()
    metadata = {'totalCells':0, 'M':X1.shape[0], 'N':X2.shape[0], 'timeStart':tic}
    print("Starting GPU Alignment...")
    path_gpu = DTWDiag_Backtrace(X1, X2, DTWDiag_fn=DTWDiag_GPU, metadata=metadata)
    metadata['time_gpu'] = time.time() - metadata['timeStart']
    print("Time GPU", metadata['time_gpu'])
    path_gpu = np.array(path_gpu)
    paths = {"path_gpu":path_gpu}
    if do_cpu:
        tic = time.time()
        path_cpu = DTW_Backtrace(X1, X2)
        elapsed = time.time() - tic
        print("Time CPU", elapsed)
        metadata["time_cpu"] = elapsed
        path_cpu = np.array(path_cpu)
        paths["path_cpu"] = path_cpu
        metadata["hist1"] = getAlignmentCellDists(path_cpu, path_gpu)['hist']
        metadata["hist2"] = getAlignmentCellDists(path_gpu, path_cpu)['hist']

    for f in ['totalCells', 'M', 'N']:
        metadata[f] = int(metadata[f])
    for f in ['XGPU', 'YGPU', 'timeStart']:
        if f in metadata:
            del metadata[f]
    json.dump(metadata, open("{}_{}_stats.json".format(filename1, prefix), "w"))
    path_gpu_arr = path_gpu.copy()
    sio.savemat(pathfilename, paths)

    path_gpu = makePathStrictlyIncrease(path_gpu)
    path_gpu *= hop_length
    path_gpu = [(row[0], row[1]) for row in path_gpu]
    path_gpu.append((x1.size, x2.size))

    if do_stretch:
        print("Stretching...")
        
        x1_stretch = pyrb.timemap_stretch(x1, sr, path_gpu)
        x3 = np.zeros((x2.size, 2))
        x3[:, 0] = x2
        x3[:, 1] = x1_stretch 
        wavfilename = "{}_{}_sync.wav".format(filename1, prefix)
        mp3filename = "{}_{}_sync.mp3".format(filename1, prefix)
        if os.path.exists(wavfilename):
            os.remove(wavfilename)
        if os.path.exists(mp3filename):
            os.remove(mp3filename)
        wavfile.write(wavfilename, sr, x3)
        subprocess.call(["ffmpeg", "-i", wavfilename, mp3filename])
        os.remove(wavfilename)

def align_corpus(foldername, do_cpu, do_stretch):
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
                align_pieces(filename1, filename2, sr, hop_length, do_mfcc=do_mfcc, do_cpu=do_cpu, do_stretch=do_stretch)
            except:
                print("ERROR")

if __name__ == '__main__':
    download_corpus("OrchestralPieces/Short")

    align_corpus("OrchestralPieces/Short", do_cpu=True, do_stretch=True)

if __name__ == '__main__2':
    hop_length = 512
    sr = 22050
    filename1 = "OrchestralPieces/Short/0_0.mp3"
    filename2 = "OrchestralPieces/Short/0_1.mp3"

    print("Doing ", filename1, filename2)
    align_pieces(filename1, filename2, sr, hop_length, do_mfcc=True, do_cpu=True, do_stretch=True)