import mne
import matplotlib.pyplot as plt
import numpy as np
from eeg_prep.utils import compute_samplepoints
from connection_complexity.data.raw_data.EDF.edf_helpers import read_edf


# edf = read_edf("f:\\data\\iEEG\\raw_ieeg\\baseline_patients\\baseline_edfs\\034_Baseline.EDF")
raw= mne.io.read_raw_edf("/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs/018_Baseline.EDF", preload=True)


# initialize nyquist and line frequency
nyq_freq = raw.info["sfreq"] // 2 - 1
line_freq = 60

# 1. filter data - bandpass [0.5, Nyquist]
l_freq = 0.5
h_freq = min(nyq_freq, 300)

# perform band-pass filtering
raw = raw.filter(l_freq=l_freq, h_freq=h_freq)