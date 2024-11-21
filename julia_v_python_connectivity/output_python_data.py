
import numpy as np
import xarray as xr
from scipy.signal.windows import dpss as sp_dpss
import os
import pickle
from mne.time_frequency.tfr import _get_nfft
from scipy.fft import fft, ifft
from mne.time_frequency import dpss_windows
from mne_connectivity.spectral.smooth import _smooth_spectra

def sinusoidal(a, f, sr, t, theta=0, DC=0):
    delta_i = 1 / sr
    f2pi = f * 2 * np.pi
    nu = np.array([DC + (a * np.sin(f2pi * i * delta_i + theta)) for i in range(t)])
    return nu


def _centered(arr, newsize):
    """Aux Function to center data."""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
#
output_path = "/media/dan/Data/git/network_miner/julia_v_python_connectivity/python_outputs"

# fully clear the output directory
import shutil
shutil.rmtree(output_path, ignore_errors=True)
os.makedirs(output_path, exist_ok=True)

t=32
sr=32
f=2
freqs=np.arange(2,16)
faverage=True
method=['coh']
mode="multitaper"

# make fake data in shape of (epoch, channel, time)
v=sinusoidal(10, f, sr, t*4, 0)
w=sinusoidal(10, f, sr, t*4, np.pi/4)
y=sinusoidal(10, f, sr, t*4, np.pi/2)
z=sinusoidal(10, f, sr, t*4, np.pi)
data = np.array([[v, w, y, z],[-v, -w, -y, -z]])

# save data to file
np.save(os.path.join(output_path,"data.npy"), data)

#### spectral_connectivity_time
n_cycles=7
mt_bandwidth = None
padding = 0 
n_jobs = 1
verbose = True


sfreq = sr
multivariate_con = False
n_signals = data.shape[1]
indices_use = np.tril_indices(n_signals, k=-1)
n_cons = len(indices_use[0])
signals_use = np.unique(np.r_[indices_use[0], indices_use[1]])
source_idx = indices_use[0].copy()
target_idx = indices_use[1].copy()
max_n_channels = len(indices_use[0])
rank = None
gc_n_lags = None
n_epochs, n_signals, n_times = data.shape


sm_times = 0
# convert kernel width in time to samples
if isinstance(sm_times, (int, float)):
    sm_times = int(np.round(sm_times * sfreq))

# convert frequency smoothing from hz to samples
sm_freqs = 0
if isinstance(sm_freqs, (int, float)):
    sm_freqs = int(np.round(max(sm_freqs, 1)))

# temporal decimation
decim = 1
if isinstance(decim, int):
    sm_times = int(np.round(sm_times / decim))
    sm_times = max(sm_times, 1)

# Create smoothing kernel
from mne_connectivity.spectral.smooth import _create_kernel
sm_kernel = "hanning"
kernel = _create_kernel(sm_times, sm_freqs, kernel=sm_kernel)


fmin = np.min(freqs)
fmax = np.max(freqs)
fmin = np.array((fmin,), dtype=float).ravel()
fmax = np.array((fmax,), dtype=float).ravel()

_f = xr.DataArray(np.arange(len(freqs)), dims=("freqs",), coords=(freqs,))
foi_s = _f.sel(freqs=fmin, method="nearest").data
foi_e = _f.sel(freqs=fmax, method="nearest").data
foi_idx = np.c_[foi_s, foi_e]
f_vec = freqs[foi_idx].mean(1)
if faverage:
    n_freqs = len(fmin)
    out_freqs = f_vec
else:
    n_freqs = len(freqs)
    out_freqs = freqs

conn = dict()
conn_patterns = dict()
for m in method:
    con_scores_dtype = np.float64
    conn[m] = np.zeros((n_epochs, n_cons, n_freqs), dtype=con_scores_dtype)
    conn_patterns[m] = None

######## Compute connectivity
call_params = dict(
    method=method,
    kernel=kernel,
    foi_idx=foi_idx,
    source_idx=source_idx,
    target_idx=target_idx,
    signals_use=signals_use,
    mode=mode,
    sfreq=sfreq,
    freqs=freqs,
    faverage=faverage,
    n_cycles=n_cycles,
    mt_bandwidth=mt_bandwidth,
    gc_n_lags=gc_n_lags,
    rank=rank,
    decim=decim,
    padding=padding,
    kw_cwt={},
    kw_mt={},
    n_jobs=n_jobs,
    verbose=verbose,
    multivariate_con=multivariate_con,
)

for epoch_idx in np.arange(n_epochs):
    print(f"   Processing epoch {epoch_idx+1} / {n_epochs} ...")

    #### _spectral_connectiviy
    eData = data[epoch_idx]
    n_cons = len(source_idx)
    eData = np.expand_dims(eData, axis=0)

    # multitaper
    ### tfr_array_multitaper
    time_bandwidth = 4.0
    decim_slice = slice(None, None, decim)
    ### _make_dpss| Returns W
    zero_mean = True
    Ws = list()

    n_taps = int(np.floor(time_bandwidth - 1))
    n_cycles_array = np.atleast_1d(n_cycles)

    for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles_array) != 1:
                this_n_cycles = n_cycles_array[k]
            else:
                this_n_cycles = n_cycles_array[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0.0, t_win, 1.0 / sfreq)

            t_centered = (t - t_win / 2.0)
            np.save(os.path.join(output_path,f"t_centered~epoch{epoch_idx}_tap{m}_freq{f}"), t_centered)

            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.0))

            # save oscillation to file
            np.save(os.path.join(output_path,f"oscillation~epoch{epoch_idx}_tap{m}_freq{f}"), oscillation)

            # Get dpss tapers
            #### dpss_windows
            N = t.shape[0]
            half_nbw = time_bandwidth / 2.0
            Kmax = n_taps
            sym = False
            norm = None
            low_bias = True
            dpss, eigvals = sp_dpss(N, half_nbw, Kmax, sym=sym, norm=norm, return_ratios=True)
            if low_bias:
                idx = eigvals > 0.9
                if not idx.any():
                    print("No tapers satisfy the low_bias constraint.")
                    idx = [np.argmax(eigvals)]
                dpss, eigvals = dpss[idx], eigvals[idx]
            tapers = dpss
            conc = eigvals

            # save dpss to file
            np.save(os.path.join(output_path,f"dpss~epoch{epoch_idx}_tap{m}_freq{f}"), dpss)
            np.save(os.path.join(output_path,f"eigvals~epoch{epoch_idx}_tap{m}_freq{f}"), eigvals)

            Wk = oscillation * tapers[m]

            # save Wk to file
            np.save(os.path.join(output_path,f"intialWk~epoch{epoch_idx}_tap{m}_freq{f}"),Wk)

            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= np.sqrt(0.5) * np.linalg.norm(Wk.ravel())

            # save Wk to file
            np.save(os.path.join(output_path,f"normalizedWk~epoch{epoch_idx}_tap{m}_freq{f}"),Wk)
            Wm.append(Wk)

        Ws.append(Wm)

    # redundant but save Ws to file
    # output of _make_dpss | Ws is a list and needs to be pickled
    with open(os.path.join(output_path,f"Ws~epoch{epoch_idx}.pkl"), 'wb') as f:
        pickle.dump(Ws, f)

    # Initialize output
    output="complex"
    epoch_data = eData
    n_freqs = len(freqs)
    n_tapers = len(Ws)
    n_epochs, n_chans, n_times = epoch_data[:, :, decim_slice].shape
    out = np.empty((n_chans, n_tapers, n_epochs, n_freqs, n_times), np.complex128)

    all_Ws = sum([list(W) for W in Ws], list())
    # save all_Ws to file
    # is a list and needs to be pickled
    with open(os.path.join(output_path,f"all_Ws~epoch{epoch_idx}.pkl"), 'wb') as f:
        pickle.dump(all_Ws, f)

    # Parallel computation 
    # in orig but unwrapped here
    # happens accross channels
    out = np.empty((n_chans, n_tapers, n_epochs, n_freqs, n_times), np.complex128)
    for ch_idx, channel in enumerate(epoch_data.transpose(1, 0, 2)):
        ## _time_frequency_loop
        ## uses Ws from above
        mode_tfl = "same"
        use_fft = True
        X_tfl = channel # shouold be shaped (1, n_times) for each channel

        # save X to file
        np.save(os.path.join(output_path,f"X_tfl~epoch{epoch_idx}_channel{ch_idx}.npy"), X_tfl)

        n_tapers = len(Ws)
        n_epochs_tfl, n_times_tfl = X_tfl[:, decim_slice].shape
        n_freqs = len(Ws[0])

        tfrs = np.zeros((n_tapers, n_epochs_tfl, n_freqs, n_times_tfl), dtype=np.complex128)

        nffts = []
        # Loops across tapers.
        for taper_idx, W in enumerate(Ws):
            # No need to check here, it's done earlier (outside parallel part)
            nfft = _get_nfft(W, X_tfl, use_fft, check=False)
            nffts.append(nfft)

            # _cwt_gen(X, W, fsize=nfft, mode=mode, decim=decim, use_fft=use_fft)
            # mode == mode_tfl == "same"
            # decim == decim_slice
            # _cwt_gen names W as Ws which overwrites the Ws from above. Renamed as wavelets for clarity
            wavelets = W

            fsize = nfft
            # precompute wavelets for given frequency range
            _, n_times_cwt = X_tfl.shape
            n_times_out = X_tfl[:, decim_slice].shape[1]
            n_freqs = len(wavelets)

            fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
            for i, Wq in enumerate(wavelets):
                fft_Ws[i] = fft(Wq, fsize)
    
            # save fft_Ws to file
            np.save(os.path.join(output_path,f"fft_Ws~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}.npy"), fft_Ws)

            tfr = np.zeros((n_freqs, n_times_out), dtype=np.complex128)

            for xidx,x in enumerate(X_tfl):
                # little x here is a single channel

                fft_x = fft(x, fsize)
                # save fft_x to file
                np.save(os.path.join(output_path,f"fft_x~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}_xidx{xidx}.npy"), fft_x)

                # Loop across wavelets
                for ii, Wq in enumerate(wavelets):
                    # ret is complex in code broken down further here to check against julia
                    # original                 
                    # ret = ifft(fft_x * fft_Ws[ii])[: n_times + Wq.size - 1]

                    tapered_fft = fft_x * fft_Ws[ii]
                    np.save(os.path.join(output_path,f"tapered_fft~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}_xidx{xidx}_wavelet{ii}.npy"), tapered_fft)

                    ret_ifft = ifft(tapered_fft)
                    np.save(os.path.join(output_path,f"ret_ifft~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}_xidx{xidx}_wavelet{ii}.npy"), ret_ifft)

                    ret = ret_ifft[: n_times + Wq.size - 1]
                    np.save(os.path.join(output_path,f"ret~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}_xidx{xidx}_wavelet{ii}.npy"), ret)

                    # mode is 'same' here so simplfying from original code
                    ret_centered = _centered(ret, n_times)
                    np.save(os.path.join(output_path,f"ret_centered~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}_xidx{xidx}_wavelet{ii}.npy"), ret_centered)
                    tfr[ii, :] = ret_centered[decim_slice]

            # _cwt_gen returns tfr but it is renamed
            # to coefs in the #loop across tapers section
            coefs = tfr
            # save coefs to file
            np.save(os.path.join(output_path,f"coefs~epoch{epoch_idx}_channel{ch_idx}_taper{taper_idx}.npy"), coefs)

            # the loop here says "loop across epochs" for
            # coefs but always a single epoch so simplifying
            # from original code I'm just going to
            # fill in tfrs with coefs
            # tfrs.shape == (3, 1, 14, 128)
            # which is (n_tapers, n_epochs_tfl, n_freqs, n_times_tfl)
            tfrs[taper_idx, 0, :, :] += coefs

        # save tfrs to file
        np.save(os.path.join(output_path,f"tfrs~epoch{epoch_idx}_channel{ch_idx}.npy"), tfrs)

        # save nffts to file at end
        # is a list and needs to be pickled
        with open(os.path.join(output_path,f"nffts~epoch{epoch_idx}_channel{ch_idx}.pkl"), 'wb') as f:
            pickle.dump(nffts, f)
    
        out[ch_idx] = tfrs

    # first dimention is epochs
    out = out.transpose(2, 0, 1, 3, 4)
    # save out to file
    np.save(os.path.join(output_path,f"out~epoch{epoch_idx}.npy"), out)

    if isinstance(n_cycles, (int, float)):
        n_cycles_array = [n_cycles] * len(freqs)

    mt_bandwidth_num = mt_bandwidth if mt_bandwidth else 4
    n_tapers = int(np.floor(mt_bandwidth_num - 1))
    weights = np.zeros((n_tapers, len(freqs), out.shape[-1]))

    for i, (f, n_c) in enumerate(zip(freqs, n_cycles_array)):
        window_length = np.arange(0.0, n_c / float(f), 1.0 / sfreq).shape[0]
        half_nbw = mt_bandwidth_num / 2.0
        tap, eigvals = dpss_windows(window_length, half_nbw, n_tapers, sym=False)
        # save tap and eigvals to file
        np.save(os.path.join(output_path,f"tap-weights~epoch{epoch_idx}_freq{f}.npy"), tap)
        np.save(os.path.join(output_path,f"eigvals-weights~epoch{epoch_idx}_freq{f}.npy"), eigvals)
        weights[:, i, :] = np.sqrt(eigvals[:, np.newaxis])

    # save weights to file
    np.save(os.path.join(output_path,f"weights~epoch{epoch_idx}.npy"), weights)

    out = np.squeeze(out, axis=0)

    psd = weights * out
    # save psd to file
    np.save(os.path.join(output_path,f"psd~epoch{epoch_idx}.npy"), psd)
    psd = psd * np.conj(psd)
    np.save(os.path.join(output_path,f"psd1~epoch{epoch_idx}.npy"), psd)
    psd = psd.real.sum(axis=1)
    np.save(os.path.join(output_path,f"psd2~epoch{epoch_idx}.npy"), psd)
    psd = psd * 2 / (weights * weights.conj()).real.sum(axis=0)
    np.save(os.path.join(output_path,f"psd3~epoch{epoch_idx}.npy"), psd)
    psd = _smooth_spectra(psd, kernel)
    np.save(os.path.join(output_path,f"psd4~epoch{epoch_idx}.npy"), psd)

    # connectivity estimation
    for s, t in zip(source_idx, target_idx):
        w_x, w_y = out[s], out[t]
        s_xy = np.sum(weights * w_x * np.conj(weights * w_y), axis=0)
        np.save(os.path.join(output_path,f"s_xy~epoch{epoch_idx}_source{s}_target{t}.npy"), s_xy)
        s_xy = s_xy * 2 / (weights * np.conj(weights)).real.sum(axis=0)
        np.save(os.path.join(output_path,f"s_xy2~epoch{epoch_idx}_source{s}_target{t}.npy"), s_xy)
        s_xy = _smooth_spectra(s_xy, kernel)
        np.save(os.path.join(output_path,f"s_xy3~epoch{epoch_idx}_source{s}_target{t}.npy"), s_xy)

        s_xx = psd[s]
        s_yy = psd[t]

        # coh
        con_num = np.abs(s_xy.mean(axis=-1, keepdims=True))
        np.save(os.path.join(output_path,f"con_num~epoch{epoch_idx}_source{s}_target{t}.npy"), con_num)
        con_den = np.sqrt(
        s_xx.mean(axis=-1, keepdims=True) * s_yy.mean(axis=-1, keepdims=True)
        )
        np.save(os.path.join(output_path,f"con_den~epoch{epoch_idx}_source{s}_target{t}.npy"), con_den)
        coh = con_num / con_den
        np.save(os.path.join(output_path,f"coh~epoch{epoch_idx}_source{s}_target{t}.npy"), coh)

        # average over frequencies
        n_foi = foi_idx.shape[0]

        # get input shape and replace n_freqs with the number of foi
        sh = list(coh.shape)
        sh[-2] = n_foi

        # compute average
        conn_f = np.zeros(sh, dtype=coh.dtype)
        for n_f, (f_s, f_e) in enumerate(foi_idx):
            f_e += 1 if f_s == f_e else f_e
            conn_f[..., n_f, :] = coh[..., f_s:f_e, :].mean(-2)
        
        # save conn_f to file
        np.save(os.path.join(output_path,f"conn_favg~epoch{epoch_idx}_source{s}_target{t}.npy"), conn_f)

