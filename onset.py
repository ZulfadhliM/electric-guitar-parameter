from scipy.signal import stft, find_peaks
import numpy as np

def detect(sig, fs, window_size=1024, hop_size=512, fft_size=None, window='hann', onset_threshold=4):

	# Calculate the STFT of the signal
	_, _, spec_data = stft(sig, nperseg=window_size, 
		noverlap=window_size-hop_size, nfft=fft_size, window=window, padded=False)

	# Get magnitude of spectrum
	spec_data = np.abs(spec_data)

	# Calculate spectral flux
	spectral_flux = np.zeros(spec_data.shape[1])
	spectral_diff = spec_data[:,1:] - spec_data[:,:-1]
	spectral_flux[1:] = np.mean(spectral_diff * (spectral_diff > 0), axis=0)

	# Normalise to standard deviation
	spectral_flux = (spectral_flux - np.mean(spectral_flux)) / np.std(spectral_flux)

	# Find peaks above onset_threshold
	idx, _ = find_peaks(spectral_flux, height=onset_threshold)

	# Convert frame to times
	onset_times = idx * float(hop_size) / fs

	return onset_times
