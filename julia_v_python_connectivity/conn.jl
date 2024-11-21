using LinearAlgebra
using FFTW
using Statistics

function sinusoidal(a, f, sr, t, theta=0, DC=0)
    delta_i = 1 / sr
    f2pi = f * 2 * π
    nu = [DC + (a * sin(f2pi * i * delta_i + theta)) for i in 0:(t-1)]
    return nu
end

# Optimized zero pad function
function zero_pad!(data_padded, data)
    data_padded[1:length(data)] .= data
    data_padded[length(data)+1:end] .= 0
end

function coh(s_xx, s_yy, s_xy)
    # Compute the numerator: absolute value of the mean of s_xy along the last dimension
    con_num = abs.(mean(s_xy, dims=ndims(s_xy)))

    # Compute the denominator: square root of the product of means of s_xx and s_yy along the last dimension
    con_den = sqrt.(mean(s_xx, dims=ndims(s_xx)) .* mean(s_yy, dims=ndims(s_yy)))

    # Calculate coherence as the element-wise division of numerator by denominator
    coh = con_num ./ con_den
    return coh
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

function _get_nfft(Ws, X)::Int
    max_len = maximum([length(Wk) for Wk in Ws])
    n = last(size(X))
    nfft = n + max_len - 1
    # @show nfft
    nfft = next_fast_len(nfft)
    return nfft
end

# Precompute FFTs of Ws
function precompute_fft_Ws(Ws, nfft)
    n_taps, n_freqs = size(Ws)
    fft_Ws = Array{ComplexF64,3}(undef, n_taps, n_freqs, nfft)
    for taper_idx = 1:n_taps
        for freq_idx = 1:n_freqs
            W = Ws[taper_idx, freq_idx]
            padded_W = zeros(ComplexF64, nfft)
            zero_pad!(padded_W, W)
            fft_Ws[taper_idx, freq_idx, :] = fft(padded_W)
        end
    end
    return fft_Ws
end

# Precompute FFTs of X
function precompute_fft_X(X, nfft)
    n_epochs, n_channels, n_times = size(X)
    fft_X = Array{ComplexF64,3}(undef, n_epochs, n_channels, nfft)
    for epoch_idx = 1:n_epochs
        for channel_idx = 1:n_channels
            x = X[epoch_idx, channel_idx, :]
            padded_x = zeros(ComplexF64, nfft)
            zero_pad!(padded_x, x)
            fft_X[epoch_idx, channel_idx, :] = fft(padded_x)
        end
    end
    return fft_X
end


# Main function to compute tfr
function compute_tfr(X, Ws, nfft)
    n_epochs, n_channels, n_times = size(X)
    n_taps, n_freqs = size(Ws)
    tfr = zeros(ComplexF64, n_epochs, n_channels, n_taps, n_freqs, n_times)

    # Precompute FFTs
    fft_Ws = precompute_fft_Ws(Ws, nfft)
    fft_X = precompute_fft_X(X, nfft)

    # Loop over tapers and frequencies
    for taper_idx = 1:n_taps
        for freq_idx = 1:n_freqs
            fft_W = fft_Ws[taper_idx, freq_idx, :]
            W_size = length(Ws[taper_idx, freq_idx])
            total_size = n_times + W_size - 1
            ret_size = total_size
            # Preallocate ret array
            ret = zeros(ComplexF64, ret_size)

            for epoch_idx = 1:n_epochs
                for channel_idx = 1:n_channels
                    fx = fft_X[epoch_idx, channel_idx, :]
                    product = fx .* fft_W
                    ret .= ifft(product)[1:ret_size]

                    # Center the result
                    start = Int(floor((ret_size - n_times) / 2)) + 1
                    end_time = start + n_times - 1
                    tfr[epoch_idx, channel_idx, taper_idx, freq_idx, :] = ret[start:end_time]
                end
            end
        end
    end
    return tfr, fft_Ws, fft_X
end

function _extend(M::Int, sym::Bool)::Tuple{Int, Bool}
    # Extend window by 1 sample if needed for DFT-even symmetry
    if !sym
        return M + 1, true
    else
        return M, false
    end
end


function _fftautocorr(x::AbstractMatrix{<:Float64})
    """
    tested vs python:
    isapprox(x_fft, py_x_fft, atol=1e-12) == true
    isapprox(py_cxy, cxy, atol=1e-12) == true
    """
    N = size(x, 2)
    use_N = next_fast_len(2 * N - 1)
    x_fft = Array{ComplexF64}(undef, size(x, 1), div(use_N, 2) + 1) # rfft returns N/2 + 1 complex numbers
    padded = zeros(Float64, size(x,1), use_N)
    padded[:, 1:N] .= x
    
    for i = 1:size(x,1)
        x_fft[i, :] = rfft(@view padded[i, :])
    end
    cxy = irfft(x_fft .* conj.(x_fft),use_N,2)[:, 1:N]    
    return cxy
end

function py_dpss(M::Int, NW::Float64, normalization_type::Int, Kmax::Int; sym::Bool=true)::Tuple{Array{Complex{Float64}, 2}, Union{Array{Float64, 1}, Float64}}
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

    known_norms = [1,2,3]
    if !(normalization_type in known_norms)
        error("normalization_type must be one of $known_norms, got $normalization_type")
    end
    if Kmax === 1
        singleton = true
    else
        singleton = false
    end
    if !(0 < Kmax <= M)
        error("Kmax must be greater than 0 and less than M")
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
    T = Tridiagonal(e, d, e)
    eigs = eigen(T)
    eigvals = eigs.values
    eigvecs = eigs.vectors
    indices = M-Kmax+1:M
    eigvals = eigvals[indices]
    eigvals = reverse(eigvals)
    eigvecs = eigvecs[end:-1:1, indices];
    windows = eigvecs[:, end:-1:1]' 
    # Correct the sign conventions
    fix_even = sum(windows[1:2:end, :], dims=2) .< 0
    for (i, f) in enumerate(fix_even)
        if f[1]
            windows[2i-1, :] *= -1
        end
    end
    thresh = max(1e-7, 1.0 / M)
    for (i, w) in enumerate(eachrow(windows[2:2:end, :]))
        idx = findfirst(x -> x^2 > thresh, w)
        if idx !== nothing && w[idx] < 0
            windows[2i, :] *= -1
        end
    end
    # compute eigenvalues. 
    dpss_rxx = _fftautocorr(windows)
    r = 4 * W * sinc.(2 * W .* nidx)
    r[1] = 2 * W
    ratios = dpss_rxx * r
    if singleton
        ratios = ratios[1]
    end

    if normalization_type != 1
        # checked vs python and it is the same
        windows .= windows ./ maximum(abs.(windows))
        if iseven(M)
            if normalization_type == 2
                correction = M^2 / (M^2 + NW)
            elseif normalization_type == 3
                s = rfft(windows[1, :])
                shift = -(1 - 1.0 / M) .* collect(1:Int(M / 2))
                s[2:end] .= s[2:end] .* (2 .* exp.(-im * π .* shift))
                correction = M / sum(real.(s))
            end
            windows .= windows .* correction
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

function compute_tapers(N::Int, n_taps::Int, freqs::AbstractArray{<:Real}, mt_bandwidth::Real, n_cycles::Int, sfreq::Int; zero_mean::Bool=true)
    n_freqs = length(freqs)
    weights = Array{Float64, 3}(undef, n_taps, n_freqs, N)
    Ws = Matrix{Vector{ComplexF64}}(undef, n_taps, n_freqs)
    sp5 = sqrt(0.5)
    # Loop over frequencies first
    for k in eachindex(freqs)
        f = freqs[k]
        t_win = n_cycles / f
        len_t = Int(ceil(t_win * sfreq))
    
        t = collect(0:1/sfreq:t_win-(t_win % (1 / sfreq) == 0 ? 1 / sfreq : 0)) # exclude last value if it fits exactly
        t_centered = t .- t_win / 2.0
    
        # Precompute oscillation and taper
        oscillation = exp.(2.0 * im * pi * f .* t_centered)
    
        taper,e = py_dpss(len_t, mt_bandwidth / 2, Kmax= n_taps, sym=false, norm=2, return_ratios=true)
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


t = 32
sr = 32
f = 2

v = sinusoidal(10, f, sr, t * 4, 0)
w = sinusoidal(10, f, sr, t * 4, π / 4)
y = sinusoidal(10, f, sr, t * 4, π / 2)
z = sinusoidal(10, f, sr, t * 4, π)

data = Array{Float64}(undef, 2, 4, 128)

data[1, :, :] = hcat(v, w, y, z)'
data[2, :, :] = hcat(-v, -w, -y, -z)';

freqs = collect(2:15) # inclusive of end 
n_freqs = length(freqs)
mt_bandwidth = 4
n_taps = floor(Int, mt_bandwidth - 1)
n_cycles = 7
sfreq = 32
zero_mean = true


ProfileView.@profview compute_tapers(size(data,ndims(data)), n_taps, freqs, mt_bandwidth, n_cycles, sfreq)

# Ws, weights = compute_tapers(size(data,ndims(data)), n_taps, freqs, mt_bandwidth, n_cycles, sfreq)

