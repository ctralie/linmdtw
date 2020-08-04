import numpy as np
import matplotlib.pyplot as plt
import warnings

def load_audio(filename, sr = 44100):
    """
    Load an audio waveform from a file.  Try to use ffmpeg
    to convert it to a .wav file so scipy's fast wavfile loader
    can work.  Otherwise, fall back to the slower librosa

    Parameters
    ----------
    filename: string
        Path to audio file to load
    sr: int
        Sample rate to use
    
    Returns
    -------
    y: ndarray(N)
        Audio samples
    sr: int
        The sample rate that was actually used
    """
    try:
        # First, try a faster version of loading audio
        from scipy.io import wavfile
        import subprocess
        import os
        FFMPEG_BINARY = "ffmpeg"
        wavfilename = "%s.wav"%filename
        if os.path.exists(wavfilename):
            os.remove(wavfilename)
        subprocess.call([FFMPEG_BINARY, "-i", filename, "-ar", "%i"%sr, "-ac", "1", wavfilename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        _, y = wavfile.read(wavfilename)
        y = y/2.0**15
        os.remove(wavfilename)
        return y, sr
    except:
        # Otherwise, fall back to librosa
        warnings.warn("Falling back to librosa for audio reading, which may be slow for long audio files")
        import librosa
        return librosa.load(filename, sr=sr)

    
def save_audio(x, sr, outprefix):
    """
    Save audio to a file

    Parameters
    ----------
    x: ndarray(N, 2)
        Stereo audio to save
    sr: int
        Sample rate of audio to save
    outprefix: string
        Use this as the prefix of the file to which to save audio
    """
    from scipy.io import wavfile
    import subprocess
    import os
    wavfilename = "{}.wav".format(outprefix)
    mp3filename = "{}.mp3".format(outprefix)
    if os.path.exists(wavfilename):
        os.remove(wavfilename)
    if os.path.exists(mp3filename):
        os.remove(mp3filename)
    wavfile.write(wavfilename, sr, x)
    subprocess.call(["ffmpeg", "-i", wavfilename, mp3filename])
    os.remove(wavfilename)

def get_DLNC0(x, sr, hop_length, lag=10, do_plot=False):
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
    
    Returns
    -------
    X: ndarray(n_win, 12)
        The DLNC0 features
    """
    from scipy.ndimage.filters import gaussian_filter1d as gf1d
    from scipy.ndimage.filters import maximum_filter1d
    import librosa
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
        import librosa.display
        plt.subplot(411)
        librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='chroma')
    norms = np.sqrt(np.sum(X**2, 0))
    if do_plot:
        plt.subplot(412)
        plt.plot(norms)
    norms = maximum_filter1d(norms, size=int(2*sr/hop_length))
    if do_plot:
        import librosa.display
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

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop size between windows
    lam: float
        The coefficient in front of the CENS features
    
    Returns
    -------
    X: ndarray(n_win, 24)
        DLNC0 features along the first 12 columns, 
        weighted CENS along the next 12 columns
    """
    import librosa
    X1 = get_DLNC0(x, sr, hop_length).T
    X2 = lam*librosa.feature.chroma_cens(y=x, sr=sr, hop_length=hop_length).T
    return np.concatenate((X1, X2), axis=1)

def get_mfcc_mod(x, sr, hop_length, n_mfcc=120, drop=20, n_fft = 2048):
    """
    Compute the mfcc_mod features, as described in Gadermaier 2019

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
    n_fft: int
        Number of fft points to use in each window
    
    Returns
    -------
    X: ndarray(n_win, n_mfcc-drop)
        The mfcc-mod features
    """
    import skimage.transform
    import librosa
    X = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, n_mfcc = n_mfcc, n_fft=n_fft, htk=True)
    X = X[drop::, :].T
    return X

def stretch_audio(x1, x2, sr, path, hop_length, refine = True):
    """
    Wrap around pyrubberband to warp one audio stream
    to another, according to some warping path

    Parameters
    ----------
    x1: ndarray(M)
        First audio stream
    x2: ndarray(N)
        Second audio stream
    sr: int
        Sample rate
    path: ndarray(P, 2)
        Warping path, in units of windows
    hop_length: int
        The hop length between windows
    refine: boolean
        Whether to refine the warping path before alignment
    
    Returns
    -------
    x3: ndarray(N, 2)
        The synchronized audio.  x2 is in the right channel,
        and x1 stretched to x2 is in the left channel
    """
    from .alignmenttools import refine_warping_path
    import pyrubberband as pyrb
    print("Stretching...")
    path_final = path.copy()
    if refine:
        path_final = refine_warping_path(path_final)
    path_final *= hop_length
    path_final = [(row[0], row[1]) for row in path_final if row[0] < x1.size and row[1] < x2.size]
    path_final.append((x1.size, x2.size))
    x3 = np.zeros((x2.size, 2))
    x3[:, 1] = x2
    x1_stretch = pyrb.timemap_stretch(x1, sr, path_final)
    x3[:, 0] = x1_stretch
    return x3