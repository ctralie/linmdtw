import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import time
import librosa
import librosa.display
import pyrubberband as pyrb
import subprocess
from scipy.io import wavfile

from scipy.ndimage.filters import gaussian_filter1d as gf1d
from scipy.ndimage.filters import maximum_filter1d
import os

FFMPEG_BINARY = "ffmpeg"

def load_audio(filename, sr = 44100):
    print("Doing crema on %s"%filename)
    wavfilename = "%s.wav"%filename
    if os.path.exists(wavfilename):
        os.remove(wavfilename)
    subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", "1", wavfilename])
    _, y = wavfile.read(wavfilename)
    y = y/2.0**15
    os.remove(wavfilename)
    return y, sr

def normalize(X):
    norms = np.sqrt(np.sum(X**2, 1))
    norms[norms == 0] = 1
    return X/norms[:, None]

def getRectifiedDiff(X):
    ret = np.zeros_like(X)
    ret[0:-1, :] = X[1::, :] - X[0:-1, :]
    ret[ret < 0] = 0
    return ret

def getSlidingWindow(X, Win, decay=True):
    Y = np.zeros((X.shape[0], X.shape[1]*Win), dtype=X.dtype)
    M = X.shape[0]-Win+1
    decays = np.linspace(0, 1, Win+1)[1::]
    decays = np.sqrt(decays[::-1])
    dim = X.shape[1]
    for k in range(Win):
        Xk =X[k:k+M, :]
        if decay:
            Xk *= decays[k]
        Y[0:M, dim*k:dim*(k+1)] = Xk
    return Y

def getDLNC0(x, sr, hop_length, lag=10, do_plot=False):
    """
    Compute decaying locally adaptive normalize C0 (DLNC0) features
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop size between windows
    lag: int
        Number of lags to include
    """
    X = np.abs(librosa.cqt(x, sr=sr, hop_length=hop_length, bins_per_octave=12))
    # Half-wave rectify discrete derivative
    #X = librosa.amplitude_to_db(X, ref=np.max)
    #X[:, 0:-1] = X[:, 1::] - X[:, 0:-1]
    X = gf1d(X, 5, axis=1, order = 1)
    X[X < 0] = 0
    # Retain peaks
    XLeft = X[:, 0:-2]
    XRight = X[:, 2::]
    mask = np.zeros_like(X)
    mask[:, 1:-1] = (X[:, 1:-1] > XLeft)*(X[:, 1:-1] > XRight)
    X[mask < 1] = 0
    # Fold into octave
    n_octaves = int(X.shape[0]/12)
    X2 = np.zeros((12, X.shape[1]), dtype=X.dtype)
    for i in range(n_octaves):
        X2 += X[i*12:(i+1)*12, :]
    X = X2
    # Compute norms
    if do_plot:
        plt.subplot(411)
        librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='chroma')
    norms = np.sqrt(np.sum(X**2, 0))
    if do_plot:
        plt.subplot(412)
        plt.plot(norms)
    norms = maximum_filter1d(norms, size=int(2*sr/hop_length))
    if do_plot:
        plt.plot(norms)
        plt.subplot(413)
        X = X/norms[None, :]
        librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='chroma')
    # Apply LNCO
    decays = np.linspace(0, 1, lag+1)[1::]
    decays = np.sqrt(decays[::-1])
    XRet = np.zeros_like(X)
    M = X.shape[1]-lag+1
    for i in range(lag):
        XRet[:, i:i+M] += X[:, 0:M]*decays[i]
    if do_plot:
        plt.subplot(414)
        librosa.display.specshow(XRet, sr=sr, x_axis='time', y_axis='chroma')
        plt.show()
    return XRet

def get_mixed_DLNC0_CENS(x, sr, hop_length, lam=0.1):
    """
    Concatenate DLNC0 to CENS
    """
    X1 = getDLNC0(x, sr, hop_length).T
    X2 = lam*librosa.feature.chroma_cens(y=x, sr=sr, hop_length=hop_length).T
    return np.concatenate((X1, X2), axis=1)

def get_mfcc_mod(x, sr, hop_length, n_mfcc=120, drop=20, finaldim=-1, n_fft = 2048):
    """
    Compute the MFCC_mod features, as described in Gadermaier 2019
    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop size between windows
    n_mfcc: int
        Number of mfcc coefficients to compute
    drop: int
        Index under which to ignore coefficients
    finaldim: int
        Resize dimension
    n_fft: int
        Number of fft points to use in each window
    """
    X = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, n_mfcc = n_mfcc, n_fft=n_fft, htk=True)
    X = X[drop::, :].T
    if finaldim > -1:
        X = skimage.transform.resize(X, (X.shape[0], finaldim), anti_aliasing=True, mode='constant')
    return X