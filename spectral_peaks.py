from scipy.fftpack import fft
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def detect(sig, fs, f0, n_harm=30, B=2e-5):
	sig *= np.hanning(len(sig))
	n_fft = len(sig) * 4 # zero pad by a factor of 4
	fs_r = fs / float(n_fft)
	cents_range = 30.0
	k = np.arange(1, n_harm+1)

	spec = np.abs(fft(sig, n_fft))

	bins = np.zeros((n_harm, 3))
	
	b = k * np.sqrt(1.0 + B * pow(k, 2.0)) * f0 / fs_r
	bins = np.multiply(b[:,np.newaxis], 
		np.tile(np.power(2.0, np.array([-1.0, 0.0, 1.0]) * cents_range / 1200.0), 
		n_harm).reshape((n_harm,3)))
	bins = np.round(bins).astype(int)

	mag_peaks = np.zeros(n_harm)
	freq_peaks = np.zeros(n_harm)
	for harm in range(n_harm):
		peak_idx = np.argmax(spec[bins[harm, 0]:bins[harm, 2]]) + bins[harm, 0]

		#interpolation
		alpha = spec[peak_idx - 1]
		beta = spec[peak_idx]
		gamma = spec[peak_idx + 1]
		num = np.log(alpha/gamma)
		den = np.log(alpha/pow(beta,2)*gamma)
		p = num / den / 2.0
		mag_peaks[harm] = beta - (alpha - gamma) * p / 4.0
		freq_peaks[harm] = fs_r * (peak_idx + p)
	f = np.arange(len(spec)) * fs_r
	return freq_peaks, mag_peaks

def estimate_inharmonicity(spectral_peak_frequencies):
	B = []
	for i in range(1, len(spectral_peak_frequencies) + 1):
		for j in range(i + 1, len(spectral_peak_frequencies) + 1):
			tmp1 = (i * spectral_peak_frequencies[j-1]) ** 2
			tmp2 = (j * spectral_peak_frequencies[i-1]) ** 2
			B.append((tmp1 - tmp2) / (pow(j,2) * tmp2 - pow(i, 2) * tmp1))
	return np.median(B)

def flatten(X, deg, target_slope=0.0):
	K = len(X)
	k = np.arange(1,K+1)
	
	X[X < 1e-15] = 1e-15
	p = np.polyfit(np.log(k), np.log(X), deg)
	est_curve = np.polyval(p, np.log(k))

	X_flatten = X / np.exp(est_curve)
	if (target_slope !=0.0):
		return X_flatten * (pow(np.exp(est_curve), -target_slope/6.0))
	return X_flatten
