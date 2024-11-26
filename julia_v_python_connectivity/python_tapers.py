import numpy as np
from scipy import linalg, fft as sp_fft
import os
from tqdm import tqdm

OUTPUT_PATH = "/media/dan/Data/git/network_mining/julia_v_python_connectivity/org_python_dbg_outputs/tapers"

def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False
    
def _fftautocorr(x):
    """Compute the autocorrelation of a real array and crop the result."""
    N = x.shape[-1]
    use_N = sp_fft.next_fast_len(2*N-1)
    x_fft = sp_fft.rfft(x, use_N, axis=-1)
    cxy = sp_fft.irfft(x_fft * x_fft.conj(), n=use_N)[:, :N]
    # Or equivalently (but in most cases slower):
    # cxy = np.array([np.convolve(xx, yy[::-1], mode='full')
    #                 for xx, yy in zip(x, x)])[:, N-1:2*N-1]
    return cxy


def dpss(M, NW, Kmax=None, sym=True, norm=None):
    if norm is None:
        norm = 'approximate' if Kmax is None else 2
    known_norms = (2, 'approximate', 'subsample')
    if norm not in known_norms:
        raise ValueError(f'norm must be one of {known_norms}, got {norm}')
    if Kmax is None:
        singleton = True
        Kmax = 1
    else:
        singleton = False
    if not 0 < Kmax <= M:
        raise ValueError('Kmax must be greater than 0 and less than M')
    if NW >= M/2.:
        raise ValueError('NW must be less than M/2.')
    if NW <= 0:
        raise ValueError('NW must be positive')
    M_orig = M
    M, needs_trunc = _extend(M, sym)
    W = float(NW) / M
    nidx = np.arange(M)

    # Here we want to set up an optimization problem to find a sequence
    # whose energy is maximally concentrated within band [-W,W].
    # Thus, the measure lambda(T,W) is the ratio between the energy within
    # that band, and the total energy. This leads to the eigen-system
    # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
    # eigenvalue is the sequence with maximally concentrated energy. The
    # collection of eigenvectors of this system are called Slepian
    # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
    # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
    # concentration
    # [see https://en.wikipedia.org/wiki/Spectral_concentration_problem]

    # Here we set up an alternative symmetric tri-diagonal eigenvalue
    # problem such that
    # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    # the main diagonal = ([M-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,M-1]
    # and the first off-diagonal = t(M-t)/2, t=[1,2,...,M-1]
    # [see Percival and Walden, 1993]
    d = ((M - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
    np.save(os.path.join(OUTPUT_PATH, f"d~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), d)
    e = nidx[1:] * (M - nidx[1:]) / 2.
    np.save(os.path.join(OUTPUT_PATH, f"e~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), e)

    # only calculate the highest Kmax eigenvalues
    w, windows = linalg.eigh_tridiagonal(
        d, e, select='i', select_range=(M - Kmax, M - 1))
    np.save(os.path.join(OUTPUT_PATH, f"w1~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), w)
    np.save(os.path.join(OUTPUT_PATH, f"windows1~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)
    w = w[::-1]
    np.save(os.path.join(OUTPUT_PATH, f"w2~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), w)

    windows = windows[:, ::-1].T
    np.save(os.path.join(OUTPUT_PATH, f"windows2~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    fix_even = (windows[::2].sum(axis=1) < 0)
    np.save(os.path.join(OUTPUT_PATH, f"fix_even~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), fix_even)
    for i, f in enumerate(fix_even):
        if f:
            windows[2 * i] *= -1
    np.save(os.path.join(OUTPUT_PATH, f"windows3~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)


    # * antisymmetric tapers should begin with a positive lobe
    #   (this depends on the definition of "lobe", here we'll take the first
    #   point above the numerical noise, which should be good enough for
    #   sufficiently smooth functions, and more robust than relying on an
    #   algorithm that uses max(abs(w)), which is susceptible to numerical
    #   noise problems)
    thresh = max(1e-7, 1. / M)
    for i, w in enumerate(windows[1::2]):
        if w[w * w > thresh][0] < 0:
            windows[2 * i + 1] *= -1
    np.save(os.path.join(OUTPUT_PATH, f"windows4~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390
    dpss_rxx = _fftautocorr(windows)
    np.save(os.path.join(OUTPUT_PATH, f"dpss_rxx~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), dpss_rxx)
    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    np.save(os.path.join(OUTPUT_PATH, f"r~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), r)
    ratios = np.dot(dpss_rxx, r)
    np.save(os.path.join(OUTPUT_PATH, f"ratios~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), ratios)

    # Deal with sym and Kmax=None
    if norm != 2:
        windows /= windows.max()
        if M % 2 == 0:
            if norm == 'approximate':
                correction = M**2 / float(M**2 + NW)
            else:
                s = sp_fft.rfft(windows[0])
                shift = -(1 - 1./M) * np.arange(1, M//2 + 1)
                s[1:] *= 2 * np.exp(-1j * np.pi * shift)
                correction = M / s.real.sum()
            np.save(os.path.join(OUTPUT_PATH, f"correction~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), correction)
            windows *= correction
    np.save(os.path.join(OUTPUT_PATH, f"windows5~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)

    # else we're already l2 normed, so do nothing
    if needs_trunc:
        windows = windows[:, :-1]

    np.save(os.path.join(OUTPUT_PATH, f"windows6~M-{M_orig}_NW-{NW}_K-{Kmax}_sym-{sym}_norm-{norm}.npy"), windows)

    return (windows, ratios)


freqs=np.arange(14, 2000)
n_cycles = 7
sfreq = 2048

for k, f in enumerate(tqdm(freqs)):
    t_win = n_cycles / float(f)
    t = np.arange(0.0, t_win, 1.0 / sfreq)
    N = t.shape[0]
    dpss(N, 2, 3, sym=False, norm=2)
