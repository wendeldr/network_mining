
import numpy as np
import os
from mne_connectivity import spectral_connectivity_time
output_path = "/media/dan/Data/git/network_miner/julia_v_python_connectivity/org_python_dbg_outputs"

def sinusoidal(a, f, sr, t, theta=0, DC=0):
    delta_i = 1 / sr
    f2pi = f * 2 * np.pi
    nu = np.array([DC + (a * np.sin(f2pi * i * delta_i + theta)) for i in range(t)])
    return nu
t=32
sr=32
f=2
noise=1000

v=sinusoidal(10, f, sr, t*4, 0)
w=sinusoidal(10, f, sr, t*4, np.pi/4)
y=sinusoidal(10, f, sr, t*4, np.pi/2)
z=sinusoidal(10, f, sr, t*4, np.pi)
# need an array shaped epoch, channel, time
data = np.array([[v, w, y, z],[-v, -w, -y, -z]])

conn = spectral_connectivity_time(
    data,
    freqs=np.arange(2,16),
    method='coh',
    sfreq=sr,
    faverage=True,
    verbose=True,
    # n_jobs=20,
    # indices=indices,
    mode="multitaper",
)

out = conn.get_data(output='dense')
np.save(os.path.join(output_path,"coh_favg_unmodified_python.npy"), out)
