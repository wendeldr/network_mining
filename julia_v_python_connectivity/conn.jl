using LinearAlgebra
using FFTW
using Statistics
using JLD2
using NPZ
using ProgressMeter
# using BenchmarkTools

function sinusoidal(a, f, sr, t, theta=0, DC=0)
    delta_i = 1 / sr
    f2pi = f * 2 * π
    nu = [DC + (a * sin(f2pi * i * delta_i + theta)) for i in 0:(t-1)]
    return nu
end

function tfr_estimate_size(n_epochs, n_channels, n_taps, n_freqs, n_times)
    element_size = 16 # bytes for ComplexF64
    total_elements = n_epochs * n_channels * n_taps * n_freqs * n_times
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function weights_estimate_size(n_taps, n_freqs, n_times)
    element_size = 8 # bytes for Float64
    total_elements = n_taps * n_freqs * n_times
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function Ws_estimate_size(n_taps, freqs, sfreq, n_cycles)
    element_size = 16 # bytes for ComplexF64
    total_elements = 0
    for k = 1:n_freqs
        f = freqs[k]
        t_win = n_cycles / f
        len_t = Int(ceil(t_win * sfreq))
        total_elements += n_taps * len_t
    end
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function fft_estimate_size(a, b, c)
    element_size = 16 # bytes for ComplexF64
    total_elements = a * b * c
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function psd_estimate_size(n_epochs, n_channels, n_freqs, n_times)
    element_size = 8 # bytes for Float64
    total_elements = n_epochs * n_channels * n_freqs * n_times
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function coh_estimate_size(n_epochs, n_channels, n_freqs)
    element_size = 8 # bytes for Float64
    total_elements = n_epochs * n_channels * n_channels * n_freqs
    total_bytes = total_elements * element_size
    total_gb = total_bytes / (1024^3) # Convert bytes to GB
    return total_gb
end

function scale_dimensions(data, n_taps, freqs, sfreq, n_cycles; print=false, max_gb=0, reserve_gb=0)
    system_mem = Sys.total_memory() / (1024^3)
    if max_gb == 0
        max_gb =  system_mem- reserve_gb # Leave X GB for other stuff
    end
    current_mem = Sys.free_memory() / (1024^3)
    if current_mem <= max_gb
        @warn "Current free memory: $current_mem GB\nDesired max: $max_gb GB\nSystem total memory: $(system_mem) GB\nMemory available is less than desired max!\nAttempting to use available memory!"
        max_gb = current_mem - reserve_gb
    end
    if(max_gb <= 0)
        error("Not enough memory available!")
    end
    
    n_epoch_org, n_channels, n_times = size(data)

    n_freqs = length(freqs)
    t_win = n_cycles / minimum(freqs)
    max_len = Int(ceil(t_win * sfreq))
    nfft = n_times + max_len - 1
    nfft = next_fast_len(nfft)
    
    data_size = Base.summarysize(data) / (1024^3)
    weights = weights_estimate_size(n_taps, n_freqs, n_times)
    Ws = Ws_estimate_size(n_taps, freqs, sfreq, n_cycles)
    fft_Ws = fft_estimate_size(n_taps, n_freqs, nfft)
    fft_X = fft_estimate_size(n_epoch_org, n_channels, nfft)
    coherence_mean = coh_estimate_size(n_epoch_org, n_channels, 1)
    
    current_mem = Sys.free_memory() / (1024^3)
    
    static_total = data_size + weights + Ws + fft_Ws + fft_X + coherence_mean
    
    if static_total >= max_gb || static_total >= current_mem
        println("Static calculations will exceed memory!")
        println("---------------------------------")
        println("Current free memory: $current_mem GB")
        println("---------------------------------")
        println("Data: $(data_size) GB")
        println("Weights: $weights GB")
        println("Ws: $Ws GB")
        println("FFT Ws: $fft_Ws GB")
        println("FFT X: $fft_X GB")
        println("Coherence Mean: $coherence_mean GB")
        println("---------------------------------")
        println("Total: $static_total GB")
        println("Exceeds maximum memory limit of $max_gb GB or current free memory")
        error("Memory limit exceeded")
    end
    n_epochs = copy(n_epoch_org)
    tfr_size = tfr_estimate_size(n_epochs, n_channels, n_taps, n_freqs, n_times)
    psd_per_epoch = psd_estimate_size(n_epochs, n_channels, n_freqs, n_times)
    coherence = coh_estimate_size(n_epoch_org, n_channels, n_freqs)
    coherence_mean_small = 0
    
    dynamic_total = tfr_size + psd_per_epoch + coherence + coherence_mean_small    
    while dynamic_total + static_total >= max_gb && n_epochs > 0
        n_epochs -= 1
        tfr_size = tfr_estimate_size(n_epochs, n_channels, n_taps, n_freqs, n_times)
        psd_per_epoch = psd_estimate_size(n_epochs, n_channels, n_freqs, n_times)
        coherence = coh_estimate_size(n_epochs, n_channels, n_freqs)
        coherence_mean_small = coh_estimate_size(n_epochs, n_channels, 1)
        dynamic_total = tfr_size + psd_per_epoch + coherence + coherence_mean_small
    end

    if n_epochs == 0
        current_mem = Sys.free_memory() / (1024^3)
        println("Can not even compute one epoch with current memory!")
        println("---------------------------------")
        println("Current free memory: $current_mem GB")
        println("System total memory: $system_mem GB")
        println("--------------Static arrays--------------")
        println("Data: $(data_size) GB")
        println("Weights: $weights GB")
        println("Ws: $Ws GB")
        println("FFT Ws: $fft_Ws GB")
        println("FFT X: $fft_X GB")
        println("Coherence Mean: $coherence_mean GB")
        println("Static Total: $static_total GB")
        println("-----Dynamically calculated arrays-----")
        println("TFR: $tfr_size GB")
        println("PSD: $psd_per_epoch GB")
        println("Coherence: $coherence GB")
        println("Coherence Mean (small): $coherence_mean_small GB")
        println("Dynamic Total: $dynamic_total GB")
        println("---------------------------------")
        println("Total: $dynamic_total + $static_total GB")
        println("Exceeds maximum memory limit of $max_gb GB")
        error("Memory limit exceeded")
    end



    if print
        current_mem = Sys.free_memory() / (1024^3)
        println("Can be computed with batches of $n_epochs epochs")
        println("total batches: $(ceil(n_epoch_org/n_epochs))")
        println("--------------Static arrays--------------")
        println("Data: $(data_size) GB")
        println("Weights: $weights GB")
        println("Ws: $Ws GB")
        println("FFT Ws: $fft_Ws GB")
        println("FFT X: $fft_X GB")
        println("Coherence Mean: $coherence_mean GB")
        println("Static Total: $static_total GB")
        println("-----Dynamically calculated arrays-----")
        println("TFR: $tfr_size GB")
        println("PSD: $psd_per_epoch GB")
        println("Coherence: $coherence GB")
        println("Coherence Mean (small): $coherence_mean_small GB")
        println("Dynamic Total: $dynamic_total GB")
        println("---------------------------------")
        println("Total: $(dynamic_total+static_total) GB")
        println("Desired max: $max_gb GB")
        println("---------------------------------")        
        println("Current free memory: $current_mem GB")
        println("System total memory: $system_mem GB")
    end
    return n_epochs
end

function tril_indices(n)::Array{Tuple{Int,Int},1}
    pairs = Array{Tuple{Int,Int},1}(undef, n * (n - 1) ÷ 2)
    q = 1
    for x in 1:n
        for y in (x+1):n
            pairs[q] = (x, y)
            q += 1
        end
    end
    return pairs
end


function next_fast_len(target::Int)::Int
    """
    Find the next fast size of input data to `fft`, for zero-padding, etc.

    Returns the next composite of the prime factors 2, 3, and 5 which is
    greater than or equal to `target`. (These are also known as 5-smooth
    numbers, regular numbers, or Hamming numbers.)

    Parameters
    ----------
    target : Int
        Length to start searching from. Must be a positive integer.

    Returns
    -------
    out : Int
        The first 5-smooth number greater than or equal to `target`.
    """
    # Precomputed Hamming numbers (5-smooth numbers) for quick lookup
    hams = [
        8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50,
        54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144,
        150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288,
        300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512,
        540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864,
        900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215, 1250, 1280,
        1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 1920,
        1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560,
        2592, 2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600,
        3645, 3750, 3840, 3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800,
        4860, 5000, 5120, 5184, 5400, 5625, 5760, 5832, 6000, 6075, 6144, 6250,
        6400, 6480, 6561, 6750, 6912, 7200, 7290, 7500, 7680, 7776, 8000, 8100,
        8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000
    ]

    if target <= 6
        return target
    end

    # Check if target is already a power of 2
    if (target & (target - 1)) == 0
        return target
    end

    # Quick lookup for small sizes
    if target <= hams[end]
        idx = searchsortedfirst(hams, target)
        return hams[idx]
    end

    # Function to compute the bit length of an integer
    bit_length(x::Int) = x <= 0 ? 0 : floor(Int, log2(x)) + 1

    match = typemax(Int)  # Initialize with maximum possible integer
    p5 = 1
    while p5 < target
        p35 = p5
        while p35 < target
            # Ceiling integer division
            quotient = cld(target, p35)
            p2 = 2^bit_length(quotient - 1)
            N = p2 * p35
            if N == target
                return N
            elseif N < match
                match = N
            end
            p35 *= 3
            if p35 == target
                return p35
            end
        end
        if p35 < match
            match = p35
        end
        p5 *= 5
        if p5 == target
            return p5
        end
    end
    if p5 < match
        match = p5
    end
    return match
end

### Tapers
function _extend(M::Int, sym::Bool)::Tuple{Int,Bool}
    # Extend window by 1 sample if needed for DFT-even symmetry
    if !sym
        return M + 1, true
    else
        return M, false
    end
end


function _fftautocorr(x::AbstractMatrix{<:Float64})::Array{Float64, 2}
    """
    tested vs python:
    isapprox(x_fft, py_x_fft, atol=1e-12) == true
    isapprox(py_cxy, cxy, atol=1e-12) == true
    """
    N = size(x, 2)
    use_N = next_fast_len(2 * N - 1)
    padded = zeros(Float64, size(x, 1), use_N)
    padded[:, 1:N] .= x
    plan = plan_rfft(padded, 2)
    x_fft = plan * padded
    cxy = irfft(x_fft .* conj.(x_fft), use_N, 2)[:, 1:N]
    return cxy
end

function py_dpss(M::Int, NW::Float64, normalization_type::Int, Kmax::Int; sym::Bool=true)::Tuple{Array{Complex{Float64},2},Union{Array{Float64,1},Float64}}
    """
    Compute the Discrete Prolate Spheroidal Sequences (DPSS).

    Parameters
    ----------
    M : Int
        Window length.
    NW : Float64
        Standardized half bandwidth corresponding to 2*NW = BW/f0 = BW*M*dt
        where dt is taken as 1.
    normalization_type : Int
        Normalization of the DPSS windows. Must be one of 1, 2, or 3.
        1: No normalization.
        2: Approximate normalization.
        3: Subsample normalization.
    Kmax : Int
        Number of DPSS windows to return. Must be less than or equal to M and greater than 0.
        If 1, return only a single window of shape (M,)
        instead of an array of windows of shape (Kmax, M).
    sym : Bool, optional
        When true (default), generates a symmetric window, for use in filter design.
        When false, generates a periodic window, for use in spectral analysis.

    return_ratios : Bool, optional
        If true, also return the concentration ratios in addition to the windows.

    Returns
    -------
    windows : Array{Float64, 2} or Array{Float64, 1}
        The DPSS windows. Will be 1D if `Kmax` is nothing.
    ratios : Array{Float64, 1} or Float64, optional
        The concentration ratios for the windows. Only returned if
        `return_ratios` evaluates to true. Will be scalar if `Kmax` is nothing.
    """
    known_norms = (1, 2, 3)
    if normalization_type ∉ known_norms
        error("normalization_type must be one of $known_norms, got $normalization_type")
    end
    if Kmax === 1
        singleton = true
    else
        singleton = false
    end
    if !(0 < Kmax <= M)
        error("Kmax must be greater than 0 and less than or equal to M")
    end
    if NW >= M / 2.0
        error("NW must be less than M/2.")
    end
    if NW <= 0
        error("NW must be positive")
    end

    M, needs_trunc = _extend(M, sym)
    W = NW / M
    nidx = collect(0:M-1)
    d = ((M - 1 .- 2 .* nidx) ./ 2.0) .^ 2 .* cos.(2pi * W)
    e = nidx[2:end] .* (M .- nidx[2:end]) ./ 2.0
    # Use SymTridiagonal for efficient eigenvalue computation
    T = SymTridiagonal(d, e)
    evals = eigvals(T, M-Kmax+1:M);
    evecs = eigvecs(T, evals);
    # Extract the largest Kmax eigenvalues and eigenvectors
    windows = evecs[:, end:-1:1]'
    # Correct sign conventions
    fix_even = sum(windows[1:2:end, :], dims=2) .< 0
    windows[1:2:end, :][fix_even[:, 1], :] .*= -1

    # # Correct signs for even-indexed windows
    thresh = max(1e-7, 1.0 / M)
    for (i, w) in enumerate(eachrow(windows[2:2:end, :]))
        idx = findfirst(x -> x^2 > thresh, w)
        if idx !== nothing && w[idx] < 0
            windows[2i, :] *= -1
        end
    end

    # Compute concentration ratios
    dpss_rxx = _fftautocorr(windows)
    r = 4 * W * sinc.(2 * W .* (nidx))
    r[1] = 2 * W
    ratios = dpss_rxx * r

    if singleton
        ratios = ratios[1]
    end
    # Apply normalization if needed
    if normalization_type != 1
        max_abs = maximum(abs, windows)
        windows ./= max_abs
        if iseven(M)
            if normalization_type == 2
                correction = M^2 / (M^2 + NW)
            elseif normalization_type == 3
                s = rfft(windows[1, :])
                shift = -(1 - 1.0 / M) .* (1:Int(M / 2))
                s[2:end] .*= 2 .* exp.(-im * π .* shift)
                correction = M / sum(real(s))
            end
            windows .*= correction
        end
    end

    if needs_trunc
        windows = windows[:, 1:end-1]
    end
    if singleton
        windows = windows[1, :]
    end
    return windows, ratios
end

function compute_tapers(N::Int, n_taps::Int, freqs::AbstractArray{<:Real}, mt_bandwidth::Real, n_cycles::Int, sfreq::Int; zero_mean::Bool=true)::Tuple{Matrix{Vector{ComplexF64}},Array{Float64,3}}
    n_freqs = length(freqs)
    weights = Array{Float64,3}(undef, n_taps, n_freqs, N)
    Ws = Matrix{Vector{ComplexF64}}(undef, n_taps, n_freqs)
    sp5 = sqrt(0.5)
    # Loop over frequencies first
    Threads.@threads for k in eachindex(freqs)
        f = freqs[k]
        t_win = n_cycles / f
        len_t = Int(ceil(t_win * sfreq))

        t = collect(0:1/sfreq:t_win-(t_win % (1 / sfreq) == 0 ? 1 / sfreq : 0)) # exclude last value if it fits exactly
        t_centered = t .- t_win / 2.0

        # Precompute oscillation and taper
        oscillation = exp.(2.0 * im * pi * f .* t_centered)

        taper, e = py_dpss(len_t, mt_bandwidth / 2, 1, n_taps, sym=false)
        weights[:, k, :] .= sqrt.(e)

        for m = 1:n_taps
            # Use @view to avoid copying taper column
            Wk = oscillation .* @view taper[m, :]

            if zero_mean  # To make it zero mean
                real_offset = mean(Wk)
                Wk .-= real_offset
            end

            # Normalize Wk
            Wk /= sp5 * norm(Wk)

            # Store Wk in preallocated Ws
            Ws[m, k] = Wk
        end
    end
    return Ws, weights
end
### end tapers


function _get_nfft(Ws::Matrix{Vector{ComplexF64}}, X::AbstractArray{<:Float64})::Int
    max_len = maximum([length(Wk) for Wk in Ws])
    n = last(size(X))
    nfft = n + max_len - 1
    # @show nfft
    nfft = next_fast_len(nfft)
    return nfft
end

# function coh(s_xx::AbstractMatrix{Float64}, s_yy::AbstractMatrix{Float64}, s_xy::AbstractMatrix{ComplexF64})::Array{Float64}
#     # Compute the numerator: absolute value of the mean of s_xy along the last dimension
#     con_num = abs.(mean(s_xy, dims=ndims(s_xy)))

#     # Compute the denominator: square root of the product of means of s_xx and s_yy along the last dimension
#     con_den = sqrt.(mean(s_xx, dims=ndims(s_xx)) .* mean(s_yy, dims=ndims(s_yy)))

#     # Calculate coherence as the element-wise division of numerator by denominator
#     coh = con_num ./ con_den
#     return coh
# end


# Precompute FFTs of Ws
function precompute_fft_Ws(Ws, nfft)
    n_taps, n_freqs = size(Ws)
    fft_Ws = zeros(ComplexF64, n_taps, n_freqs, nfft) # preallocated padded array
    
    # tried threading but it was slightly slower
    for taper_idx = 1:n_taps
        for freq_idx = 1:n_freqs
            # Ws are different lengths
            fft_Ws[taper_idx, freq_idx, 1:length(Ws[taper_idx, freq_idx])] .= Ws[taper_idx, freq_idx]
        end
    end
    # plan fft!
    p = plan_fft!(fft_Ws, 3)
    return p * fft_Ws
end

# Precompute FFTs of X
function precompute_fft_X(X, nfft)
    n_epochs, n_channels, n_times = size(X)
    fft_X = zeros(ComplexF64,n_epochs, n_channels, nfft)
    fft_X[:,:,1:n_times] .= data
    p = plan_fft!(fft_X, 3)
    return p * fft_X
end

function compute_tfr!(tfr::Array{ComplexF64, 5}, fft_X::Array{ComplexF64, 3}, fft_Ws::Array{ComplexF64, 3}, Ws_lengths::Array{Int64, 2})
    batch_size, n_channels, nfft = size(fft_X)
    n_taps, n_freqs, _ = size(fft_Ws)
    _, _, _, _, n_times = size(tfr)
    
    # Precompute sizes, start_indices, and end_indices
    sizes = n_times .+ Ws_lengths .- 1
    start_indices = floor.(Int, (sizes .- n_times) ./ 2) .+ 1
    end_indices = start_indices .+ n_times .- 1
    
    nthreads = Threads.nthreads()
    temp_arrays = [Array{ComplexF64}(undef, batch_size, n_channels, nfft) for _ in 1:nthreads]
    fft_plans = [plan_ifft!(temp_arrays[i], 3) for i in 1:nthreads]
    
    # Thread over frequencies
    @inbounds @showprogress desc="Computing TFRs" Threads.@threads for freq_idx = 1:n_freqs
        thread_id = Threads.threadid()
        temp = temp_arrays[thread_id]
        ifft_plan = fft_plans[thread_id]
        
        # Loop over tapers
        for taper_idx = 1:n_taps
            fft_W = fft_Ws[taper_idx, freq_idx, :]  # Current fft_W
            Ws_length = Ws_lengths[taper_idx, freq_idx]
            ret_size = n_times + Ws_length - 1
            
            # Compute start and end indices for slicing
            start = start_indices[taper_idx, freq_idx]
            end_time = end_indices[taper_idx, freq_idx]
            
            # Compute the product and inverse FFT in-place
            temp .= fft_X .* reshape(fft_W, 1, 1, nfft)  # Broadcasting over first two dims
            temp .= ifft_plan * temp  # In-place inverse FFT
            
            # Assign the centered result to tfr
            tfr[:, :, taper_idx, freq_idx, :] .= temp[:, :, start:end_time]
        end
    end
    
    return tfr
end

function compute_psd!(psd_per_epoch::Array{Float64,4}, tfrs::Array{ComplexF64,5}, weights::Array{Float64,3}, normalization::Array{Float64, 2})::Array{Float64,4}
    batch_size, n_channels, n_tapers, n_freqs, n_times = size(tfrs)

    nthreads = Threads.nthreads()
    psd_arrays = [Array{ComplexF64}(undef, n_tapers, n_freqs, n_times) for _ in 1:nthreads]
    psd_sums = [Array{Float64}(undef, n_freqs, n_times) for _ in 1:nthreads]

    @inbounds @showprogress desc="Computing epoch(s) PSD(s)..." for idx = 1:(batch_size * n_channels)
        thread_id = Threads.threadid()
        psd = psd_arrays[thread_id]
        psd_sum = psd_sums[thread_id]

        # Compute epoch_idx and channel_idx from idx
        epoch_idx = div(idx - 1, n_channels) + 1
        c_idx = mod(idx - 1, n_channels) + 1

        # Extract the current tfr slice
        tfr_view = @view tfrs[epoch_idx, c_idx, :, :, :]

        # Perform the element-wise multiplication
        @. psd = weights * tfr_view

        # Compute the squared magnitude
        @. psd = psd * conj(psd)

        # Sum across the first dimension (tapers)
        psd_sum .= 0.0
        @inbounds for t = 1:n_tapers
            @views psd_sum .= psd_sum .+ real(psd[t, :, :])
        end

        # Apply the normalization
        @. psd_sum = psd_sum * normalization

        # Update the psd_per_epoch array
        psd_per_epoch[epoch_idx, c_idx, :, :] .= psd_sum
    end
    return psd_per_epoch
end

function compute_coh_mean!(coherence::Array{Float64,4}, tfrs::Array{ComplexF64,5}, pairs::Vector{Tuple{Int64, Int64}}, psd_per_epoch::Array{Float64,4}, weights_squared::Array{Float64,3}, normalization::Array{Float64, 2})::Array{Float64,4}
    batch_size, n_channels, n_taps, n_freqs, n_times = size(tfr)
    n_pairs = length(pairs)

    nthreads = Threads.nthreads()
    temp_arrays = [Array{ComplexF64,3}(undef, n_taps, n_freqs, n_times) for _ in 1:nthreads]
    @inbounds @showprogress desc = "Computing Coherence..." Threads.@threads for idx in 1:batch_size*n_pairs
        thread_id = Threads.threadid()
        temp = temp_arrays[thread_id]

        # Calculate the epoch index and pair index
        epoch_idx = div(idx - 1, n_pairs) + 1
        pair_idx = mod(idx - 1, n_pairs) + 1
        x, y = pairs[pair_idx]
        # println("Epoch: $epoch_idx, Pair: ($x, $y)")
        # Now perform your operations
        w_x = @view tfrs[epoch_idx, x, :, :, :]
        w_y = @view tfrs[epoch_idx, y, :, :, :]
        temp .= weights_squared .* w_x .* conj.(w_y)
        s_xy = dropdims(sum(temp, dims=1),dims=1)  # sum over tapers
        s_xy .*= normalization

        s_xx = @view psd_per_epoch[epoch_idx, x, :, :]
        s_yy = @view psd_per_epoch[epoch_idx, y, :, :]

        # Compute the numerator: absolute value of the mean of s_xy along the last dimension
        con_num = abs.(mean(s_xy, dims=2))

        # Compute the denominator: square root of the product of means of s_xx and s_yy along the last dimension
        con_den = sqrt.(mean(s_xx, dims=2) .* mean(s_yy, dims=2))

        # Calculate coherence as the element-wise division of numerator by denominator
        coh_value = con_num ./ con_den
        # coh_value = coh(s_xx, s_yy, s_xy[1, :, :])

        # Copy to symmetric position
        coherence[epoch_idx, y, x, :] .= coh_value
    end
    return mean(coherence, dims=ndims(coherence))
end


outputpath = "/media/dan/Data/git/network_mining/connectivity/julia_test/"
data = npzread("/media/dan/Data/git/network_mining/connectivity/julia_test/034_input.npy")
sfreq = 2048
freqs = collect(14:250)
zero_mean = true
n_freqs = length(freqs)
mt_bandwidth = 4
n_taps = floor(Int, mt_bandwidth - 1)
n_cycles = 7
n_epochs, n_channels, n_times = size(data)

batch_size = scale_dimensions(data, n_taps, freqs, sfreq, n_cycles, print=true, reserve_gb=60)
total_batches = ceil(Int,n_epochs / batch_size)
if batch_size != n_epochs
    println("Data is too big for one pass!\nData will be computed in batches of $batch_size epochs. Total batches: $(total_batches)")
end


println("Making tapers...")
Ws, weights = compute_tapers(n_times, n_taps, freqs, mt_bandwidth, n_cycles, sfreq)
weights_squared = weights .^ 2
normalization = 2 ./ sum(real(weights .* conj(weights)), dims=1);
small_norm = dropdims(normalization; dims=1)

nfft = _get_nfft(Ws, data)

println("Precomputing FFTs of tapers and data...")
fft_Ws = precompute_fft_Ws(Ws, nfft);
fft_X = precompute_fft_X(data, nfft);
println("Done!")

# save(joinpath(outputpath, "034_pretasks.jld2"), "Ws", Ws, "weights", weights, "fft_Ws", fft_Ws, "fft_X", fft_X, "normalization", normalization)

Ws_lengths = [length(Wk) for Wk in Ws]

println("Preparing for computation...")
pairs = tril_indices(n_channels)
n_pairs = length(pairs)
tfr = Array{ComplexF64,5}(undef, batch_size, n_channels, n_taps, n_freqs, n_times);
psd_per_epoch = Array{Float64,4}(undef, batch_size, n_channels, n_freqs, n_times);
coherence = Array{Float64,4}(undef, batch_size, n_channels, n_channels, n_freqs)
coherence_mean = Array{Float64,4}(undef, n_epochs, n_channels, n_channels, 1)
for b = 1:total_batches
    println("Batch $(Int(b))/$(Int(total_batches)) ...")
    start_idx = Int((b - 1) * batch_size + 1)
    end_idx = Int(min(b * batch_size, n_epochs))
    data_batch = @view fft_X[start_idx:end_idx, :, :];
    compute_tfr!(tfr, data_batch, fft_Ws, Ws_lengths);
    compute_psd!(psd_per_epoch, tfr, weights, small_norm);
    psd_per_epoch = compute_psd(batch_size, n_channels, n_freqs, n_times, tfrs, weights, normalization)
    coherence_mean[start_idx:end_idx, :, :, :] .= compute_coh_mean!(coherence, tfr, pairs, psd_per_epoch, weights_squared, small_norm)
end
println("Saving...")
save(joinpath(outputpath, "034_coherence.jld2"), "coherence_mean", coherence_mean)
println("Done saving!")

