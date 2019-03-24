import numpy as np
from librosa.util.utils import pad_center
import matplotlib.pyplot as plt

def detect(sig, fs, onset_times=[0.0], window_size=1024, min_freq=65, threshold=0.2):
	max_lag = int(np.ceil(fs / min_freq))

	# YIN
	d = np.zeros(max_lag-1)
	pitches = np.zeros(len(onset_times))

	for i in range(len(onset_times)):
		for lag in range(max_lag-1):
			sample = int(round(onset_times[i] * fs))
			d[lag] = np.sum((sig[sample:sample+window_size]-sig[sample+lag+1:sample+window_size+lag+1])**2)
		d = d / np.cumsum(d) * np.arange(1, max_lag) # normalised

		idx_below_threshold = np.where(d < threshold)[0]
		if (len(idx_below_threshold) == 0):
			continue
		stop_at = np.where(np.diff(idx_below_threshold) > 1)[0].astype(int)
		if (len(stop_at)==0):
			idx_min = np.argmin(d[idx_below_threshold])
		else:
			search_range = idx_below_threshold[:stop_at[0]]
			idx_min = np.argmin(d[search_range])
		idx = idx_below_threshold[idx_min]

		# interpolation
		num = d[idx-1] - d[idx+1]
		den = d[idx-1] - 2 * d[idx] + d[idx+1]
		if (den != 0):
			pitches[i] = fs / (idx + num / den / 2)

	return pitches
