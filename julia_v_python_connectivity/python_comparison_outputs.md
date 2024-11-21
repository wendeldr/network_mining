/home/dan/miniconda3/envs/sourcesink/lib/python3.12/site-packages/mne/time_frequency/tfr.py

line 282

```python
out_path = "/media/dan/Data/git/network_miner/mining/images/oscillations_and_tapers/python"
import matplotlib.pyplot as plt
import os
for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles) != 1:
                this_n_cycles = n_cycles[k]
            else:
                this_n_cycles = n_cycles[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0.0, t_win, 1.0 / sfreq)
            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.0))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(oscillation.real)
            ax[0].set_title("Real")
            ax[0].grid()
            ax[1].plot(oscillation.imag)
            ax[1].set_title("Imaginary")
            ax[1].grid()
            plt.savefig(os.path.join(out_path,f"tap{m+1:03}_freq{k+1:03}_oscillation_py.png"))
            plt.close()


            # Get dpss tapers
            tapers, conc = dpss_windows(
                t.shape[0], time_bandwidth / 2.0, n_taps, sym=False
            )
            fig, ax = plt.subplots(1,len(tapers))
            for i in range(len(tapers)):
                taper = tapers[i]
                ax[i].plot(taper)
                ax[i].set_title(f"Taper {i+1}")
                ax[i].grid()
            plt.savefig(os.path.join(out_path,f"freq{k+1:03}_tapers_py.png")) # will overwrite but tapers don't change
            plt.close()

            Wk = oscillation * tapers[m]
            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= np.sqrt(0.5) * np.linalg.norm(Wk.ravel())
            fig, ax = plt.subplots(1,2)
            ax[0].plot(Wk.real)
            ax[1].plot(Wk.imag)
            ax[0].set_title("Real")
            ax[1].set_title("Imaginary")
            ax[0].grid()
            ax[1].grid()
            plt.savefig(os.path.join(out_path,f"tap{m+1:03}_freq{k+1:03}_wk_py.png"))
            plt.close()

```

Julia vs python thoughts:

in general the shapes are similar. as the frequency increases the taper, oscilations, and wk differ particularly at the end of the frequencies. I can't change this without somehow exactly duplicating python. not sure it is important as the changes are minor

---
TFR array

Within julia the first implmentation matches the second optimized julia implementation


Python vs Julia thoughts:

seems pretty similar with random differences.

---
Weights array

/home/dan/miniconda3/envs/sourcesink/lib/python3.12/site-packages/mne_connectivity/spectral/time.py
line 799

weights of julia vs python seem exactly the same

---
