import numpy as np
import os
import pandas as pd
from connection_complexity.data.raw_data.EDF.edf_helpers import read_edf
from eeg_prep.preprocess.ieeg import preprocess_ieeg
from mne_connectivity import spectral_connectivity_time
from mne import make_fixed_length_epochs

mapping_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
ilae_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
bad_channels_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/data/bad_ch_review.xlsx"
edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs"
output_path = "/media/dan/Data/git/network_miner/connectivity/output"

pid_source_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"


dur_msec = 500
dur_sec = dur_msec / 1000
overlap_msec = 0
overlap_sec = overlap_msec / 1000


# get metadata
mappings = pd.read_csv(mapping_path)
ilae = pd.read_csv(ilae_path)
bad_channels = pd.read_excel(bad_channels_path)
bad_channels["use"] = bad_channels["use"].fillna(1)
bad_channels["use2"] = bad_channels["use2"].fillna(1)
bad_channels["use"] = bad_channels["use"].astype(bool)
bad_channels["use2"] = bad_channels["use2"].astype(bool)

# OR bad_channel columns
bad_channels["bad_channel"] = ~(bad_channels["use"] & bad_channels["use2"])

# for each patient in mappings, find the corresponding ilae number. The patient may not be in the ilae dataset but has a designation of seizureFree or not.
# if the patient is not in the ilae dataset, then use the seizureFree column to determine the ilae number where -1 is seizureFree and 100 is not seizureFree
ilae_numbers = {}
for pid in mappings["pid"].unique():
    if pid in ilae["patient"].values:
        ilae_numbers[pid] = ilae[ilae["patient"] == pid]["ilae"].values[0]
    else:
        if mappings[mappings["pid"] == pid]["seizureFree"].values[0] == True:
            ilae_numbers[pid] = -1
        else:
            ilae_numbers[pid] = 100

# now we have a dictionary of ilae numbers for each patient. Fill in the mappings dataframe with these numbers which has multiple rows for each patient
ilae_list = []
for pid in mappings["pid"]:
    ilae_list.append(ilae_numbers[pid])
mappings["ilae"] = ilae_list


# Perform the merge as before
mappings = mappings.merge(
    bad_channels[['pid', 'ch', 'bad_channel']],
    left_on=['pid', 'electrode'],
    right_on=['pid', 'ch'],
    how='left'
)

# Drop the 'ch' column if needed
mappings = mappings.drop(columns=['ch'])

# Fill NaN values in 'bad_channel' with 0
mappings['bad_channel'] = mappings['bad_channel'].fillna(0)


################
# get 3 random pids with ilae 1-2 
df = pd.read_csv(pid_source_path)
df = df[df["ilae"] < 3]
df = df.sample(3, random_state=14)
pidsToUse = df['patient'].unique()


##########
files = os.listdir(edf_path)
for edf_file in files:
    file = os.path.join(edf_path, edf_file)
    pid = int(edf_file.split("_")[0])
    if pid not in pidsToUse:
        continue
    try:
        print(f"Processing {file}")
        patient_info = mappings[mappings["pid"] == pid]
        raw = read_edf(file, preload=True)
        sfreq = raw.info["sfreq"]

        raw.info["line_freq"] = 60

        raw = preprocess_ieeg(raw) 

        # remove bad channels
        raw = raw.drop_channels(patient_info[patient_info["bad_channel"] == 1]["electrode"].values)

        # average reference
        raw = raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

        # make epochs
        epochs = make_fixed_length_epochs(raw, duration=dur_sec,overlap=overlap_sec, preload=True)
        methods = ['coh',"pli","wpli","plv"]
        freqs = np.arange(14,sfreq//2)
        conn = spectral_connectivity_time(
            epochs.get_data(),
            freqs=freqs,
            method=methods,
            sfreq=sfreq,
            faverage=True,
            verbose=True,
            n_jobs=27,
            # indices=indices,
            mode="multitaper",
        )
        for i,m in enumerate(methods):
            out = conn[i].get_data(output='dense')
            # save to file
            np.save(os.path.join(output_path, f"{pid:03}_{m}_winmsec-{dur_msec:06}_overlap-{overlap_msec:06}.npy"), out)
    except Exception as e:
        with open(os.path.join(output_path,"error_log.txt"), "a") as f:
            f.write(f"Error processing {file}\n")
            f.write(str(e))
            f.write("\n============")
        print(f"Error processing {file}")
        print(e)
        print('=============')
        continue