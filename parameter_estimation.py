import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def log_correlation(X, fs, f0, resolution=0.01):
	k = np.arange(1, len(X) + 1)
	T = round(1.0 / f0 * fs)
	tau = np.arange(0, T, resolution)
	R = np.dot(np.log(X ** 2),np.cos(np.multiply.outer(2 * np.pi * k, tau/T)))

	# return half
	half_len = int(len(tau)//2)

	return tau[:half_len], R[:half_len]

def detect_pickup_pluck_positions(X, fs, f0, threshold=-10, pos_min=0.03846, pos_max=0.2769):

	tau, R = log_correlation(X, fs, f0)

	# Detect two minima locations
	locs, _ = find_peaks(-R, height=-threshold)

	# Sort locations according to amplitude of valleys
	sorted_idx = np.argsort(R[locs])
	locs = locs[sorted_idx]
	pks = R[locs]

	# Convert locations to ratios
	locs_sample = locs
	locs = tau[locs] / tau[-1] / 2.0

	# Remove valleys outside of search range
	idx = np.where(np.logical_and(locs>pos_min, locs<pos_max))[0]
	pks = pks[idx]
	locs = locs[idx]

	# Only analyse two lowest valleys
	if len(locs) > 2:
		locs = locs[:2]

	# Remove 2nd lowest valley if it is less than 40% than the lowest valley
	if len(locs) is not 1:
		if ((pks[1] / pks[0]) < 0.6):
			locs = locs[:1]

	if len(locs) == 1:
		locs_sample[0]
		est_R = R[locs_sample[0]] * 0.8 # 80 pct of lowest valley

		# search left
		locs_1 = locs_sample[0]
		found = False
		while found is False:
			if (R[locs_1] - est_R >= 0):
				found = True
			else:
				locs_1 -= 1

		# search right
		locs_2 = locs_sample[0]
		found = False
		while found is False:
			if (R[locs_2] - est_R >= 0):
				found = True
			else:
				locs_2 += 1	

		# Convert back to ratios
		locs = np.array([locs_1, locs_2])
		locs = tau[locs] / tau[-1] / 2.0

	return locs
