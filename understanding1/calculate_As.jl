using LinearAlgebra
using Base.Threads
using ProgressMeter
using JLD2
using PythonCall
using PyMNE


edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs/034_Baseline.EDF"

raw = PyMNE.io.read_raw_edf(edf_path)
sfreq = pyconvert(Any,raw.info["sfreq"])
eeg_channels = [ch for ch in pyconvert(Any,raw.ch_names) if startswith(ch,"EEG ")]


raw = PyMNE.io.read_raw_edf(edf_path, preload=true, include=pylist(eeg_channels), verbose=false)
raw = raw.filter(l_freq=0.5, h_freq=300, verbose=false)
data_py = raw.get_data()

data = pyconvert(Matrix,data_py)' # channel x time -> time x channel

# Parameters for window and step sizes in milliseconds
WINSIZE_MSEC = 500  # Window size in ms
STEPSIZE_MSEC = 500  # Step size in ms

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
sample_points = compute_samplepoints(winsize_samps, stepsize_samps, size(data, 1))
n_wins = length(sample_points)

println("Initializing storage for A matrices...")
# Initialize storage for A matrices
A_mats = zeros(size(data, 2), size(data, 2), n_wins)

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
    data_window = data[start_idx:end_idx, :]'
    A_mats[:, :, idx] = compute_statelds_func(data_window)
    next!(progress)  # Update the progress bar
end
finish!(progress)

println("Saving A matrices...")
save("034_Baseline_As_01024_00001.jld2", "A_mats", A_mats)
