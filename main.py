import soundfile as sf
import onset
import pitch
import spectral_peaks
import parameter_estimation
import numpy as np

sig, fs = sf.read("../electric-guitar-samples/string-2/1/03.wav")

# Detect times for each guitar pluck
hop_size = 512
onset_times = onset.detect(sig, fs)

# 
f0 = pitch.detect(sig, fs, onset_times)[0]

# Analyse plucked string
sample = int(round(onset_times[0] * fs))
period = 10
length = int(period * (1.0 / f0 * fs))
sig = sig[sample:sample+length]

# Extract spectral peaks
f, _ = spectral_peaks.detect(sig, fs, f0)
B = spectral_peaks.estimate_inharmonicity(f)
f, X = spectral_peaks.detect(sig, fs, f0, B=B)
X_flat = spectral_peaks.flatten(X, deg=0)

R1, R2 = parameter_estimation.detect_pickup_pluck_positions(X_flat, fs, f0)
