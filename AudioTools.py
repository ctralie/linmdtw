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
    wavfilename = "%s.wav"%filename
    if os.path.exists(wavfilename):
        os.remove(wavfilename)
    subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", "1", wavfilename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    _, y = wavfile.read(wavfilename)
    y = y/2.0**15
    os.remove(wavfilename)
    return y, sr

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

def stretch_audio(x1, x2, sr, path, hop_length, outprefix):
    """
    Wrap around pyrubberband to warp the audio
    Parameters
    ----------
    x1: ndarray
        First audio stream
    x2: ndarray
        Second audio stream
    sr: int
        Sample rate
    path: ndarray(M, 2)
        Warping path, in units of windows
    hop_length: int
        The hop length between windows
    outprefix: string
        The prefix of the output file to which to save the result
    """
    from AlignmentTools import makePathStrictlyIncrease
    print("Stretching...")
    #path = makePathStrictlyIncrease(path)
    path *= hop_length
    path = [(row[0], row[1]) for row in path]
    path.append((x1.size, x2.size))
    x3 = np.zeros((x2.size, 2))
    x3[:, 1] = x2
    x1_stretch = pyrb.timemap_stretch(x1, sr, path)
    x3[:, 0] = x1_stretch
    wavfilename = "{}.wav".format(outprefix)
    mp3filename = "{}.mp3".format(outprefix)
    if os.path.exists(wavfilename):
        os.remove(wavfilename)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    wavfile.write(wavfilename, sr, x3)
    subprocess.call(["ffmpeg", "-i", wavfilename, mp3filename])
    os.remove(wavfilename)



def test_sync():
    import scipy.io as sio
    hop_length = 512
    sr = 22050
    filename1 = "OrchestralPieces/Long/2_0.mp3"
    x1, sr = load_audio(filename1, 22050)
    filename2 = "OrchestralPieces/Long/2_1.mp3"
    x2, sr = load_audio(filename2, 22050)
    path = sio.loadmat("OrchestralPieces/Long/2_0.mp3_chroma_path.mat")['path_gpu']
    hop_length = 512
    outprefix = "synced"
    stretch_audio(x1, x2, sr, path, hop_length, outprefix, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

if __name__ == '__main__':
    test_sync()