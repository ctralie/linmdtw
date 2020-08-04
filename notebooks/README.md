# Notebooks

These notebooks show common use cases for this library.  The BasicUsage notebook only requires the dependencies in requirements.txt, while the AudioUsage library also requires what's in requirements_audio.txt.  The latter will also run much faster if you are using pycuda.

## Web audio examples

Annoyingly, github's Jupyter viewer does not allow replay of audio using HMTL5.  Therefore, if you want to simply listen to the precomputed results in the AudioUsage example, they are below

### Vivalid's Spring

Before we apply the computed warping path, let's compare the first 40 seconds of the two audio clips side by side. We'll put the first one in the left ear and the second one in the right ear. The one on the left goes faster than the one on the right, but it starts later. Because of this, they are in sync for a brief moment, but the left one then overtakes the right one for the rest of it.
<a href = "unsync0.mp3">Click here</a> to listen to the audio file <code>unsync0.mp3</code>

Let's now apply the computed warping path to see how the alignment went. This library wraps arround the pyrubberband library, which we can use to stretch the audio in x1 to match x2, according to this warping path. The method <code>stretch_audio</code> returns a stereo audio stream with the resulting stretched version of x1 in the left ear and the original version of x2 in the right ear. Let's save the first 30 seconds of this to disk and listen to it
<a href = "sync0.mp3">Click here</a> to listen to the audio file <code>sync0.mp3</code>

### Schubert's Unfinished Symphony

We now show one more example with a 45 second clip from Schubert's Unfinished Symphony (short clip index 5 in the paper corpus)
<a href = "sync5.mp3">Click here</a> to listen to the audio file <code>sync5.mp3</code>