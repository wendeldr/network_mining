using LinearAlgebra
using Base.Threads
using ProgressMeter
using JLD2
using EDF
using DSP

function is_eeg_signal(signal)
    return startswith(signal.header.label, "EEG ")
end

println("Reading EDF file...")
edf = EDF.read("/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs/034_Baseline.EDF")

sfreq = edf.signals[1].header.samples_per_record / edf.header.seconds_per_record
# Decode only EEG signals (where the label starts with "EEG ")
signals = [EDF.decode(signal) for signal in edf.signals if signal isa EDF.Signal{Int16} && is_eeg_signal(signal)]
arr = hcat(signals...) # time by channels

# filters
k = 3.3
numtaps = ceil(Int, k * sfreq / 0.5)
# numtaps = 13517        # Filter length (number of taps)
cutoff = [0.25, 337.5] # Cutoff frequencies in Hz (-6 dB points)

# Design the FIR filter using firwin
# h = firwin(numtaps, cutoff; ftype=:bandpass, window=hamming(numtaps), fs=sfreq)

# Apply zero-phase filtering using filtfilt
# x is your input signal
# y = filtfilt(h, 1, x)

# filtered_data = [filtfilt(h, 1, arr[:, ch]) for ch in 1:size(arr, 2)]
# arr_filtered = hcat(filtered_data...)

nyq_freq = sfreq / 2
line_freq = 60
l_freq = 0.5
h_freq = min(nyq_freq, 300) 

# bandpass filter
b_bandpass = digitalfilter(Bandpass(l_freq, h_freq; fs=sfreq), FIRWindow(hamming(numtaps, zerophase=true); attenuation=53))
filtered_data = [filt(b_bandpass, arr[:, ch]) for ch in 1:size(arr, 2)]
arr_filtered = hcat(filtered_data...)

# Parameters for window and step sizes in milliseconds
const WINSIZE_MSEC = 500  # Window size in ms
const STEPSIZE_MSEC = 500  # Step size in ms

# Convert window and step sizes to samples
winsize_samps = Int(round(WINSIZE_MSEC * (sfreq / 1000)))
stepsize_samps = Int(round(STEPSIZE_MSEC * (sfreq / 1000)))

# Function to compute sample points based on window and step sizes
function compute_samplepoints(winsize_samps, stepsize_samps, total_samples)
    sample_points = []
    for i in 1:stepsize_samps:(total_samples - winsize_samps)
        push!(sample_points, (i, i - 1 + winsize_samps))
    end
    return sample_points
end

# Compute sample points for the entire signal
sample_points = compute_samplepoints(winsize_samps, stepsize_samps, size(arr, 1))
n_wins = length(sample_points)

println("Initializing storage for A matrices...")
# Initialize storage for A matrices
A_mats = zeros(size(arr, 2), size(arr, 2), n_wins)

# Function to compute A matrix for each data window (channels by time)
function compute_statelds_func(data_window)
    X = data_window[:, 1:end-1]
    Y = data_window[:, 2:end]
    return Y * pinv(X)
end

println("Computing A matrices...")
# Initialize a progress bar
progress = Progress(n_wins, desc="Computing A matrices")

# Parallel computation of A matrices using threading
@threads for idx in 1:n_wins
    start_idx, end_idx = sample_points[idx]
    data_window = arr[start_idx:end_idx, :]'
    A_mats[:, :, idx] = compute_statelds_func(data_window)
    next!(progress)  # Update the progress bar
end
finish!(progress)

println("Saving A matrices...")
save("034_Baseline_As_01024_00001.jld2", "A_mats", A_mats)
