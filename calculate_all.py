from eeg_prep.network import state_lds_array
from eeg_prep.preprocess.ieeg import preprocess_ieeg
from eeg_prep.utils import compute_samplepoints
from connection_complexity.data.raw_data.EDF.edf_helpers import read_edf
from eeg_prep.metrics.sourcesink import state_sourcesink_array

import numpy as np
import glob
import os
import h5py
import pandas as pd

from tqdm import tqdm

WINSIZE_MSEC = 500
STEPSIZE_MSEC = .5


def compute_A(
    raw,
    winsize_samps,
    stepsize_samps,
):
    # Compute the state-space model
    data = raw.get_data()

    A_mats = state_lds_array(
        data, winsize=winsize_samps, stepsize=stepsize_samps, l2penalty=0, progressbar=True
    )

    # calculate the error of the A in reconstructing the data
    window_indexs = compute_samplepoints(winsize_samps, stepsize_samps, raw.n_times)
    channels = raw.ch_names
    # initialize windowed data
    n_wins = window_indexs.shape[0]
    # windows = np.zeros((winsize_samps, len(channels), n_wins))
    # errors = np.zeros(n_wins)
    # for idx in range(n_wins):
    #     windows[:, :, idx] = data[:, window_indexs[idx, 0] : window_indexs[idx, 1]].T
    #     w = windows[:, :, idx]
    #     A = A_mats[..., idx]
    #     w_hat = np.dot(A, w.T).T
    #     errors[idx] = np.linalg.norm(w - w_hat)

    windows = np.zeros((3,3,3))
    errors = np.zeros(3)

    # filter bad A's based on median * 100 (arbitrary but our outliers are very large)
    A_medians = []
    for x in range(A_mats.shape[-1]):
        A_medians.append(np.median(np.abs(A_mats[..., x])))

    A_thresh = np.median(A_medians) * 100
    A_mask = []
    for x in range(A_mats.shape[-1]):
        if A_medians[x] > A_thresh:
            A_mask.append(False)
        else:
            A_mask.append(True) # keep good A's
    A_mask = np.array(A_mask)

    A_mean = np.mean(A_mats[..., A_mask], axis=-1, keepdims=True)
    return A_mats, A_medians, A_thresh, A_mask, A_mean, window_indexs, windows, errors


def save_h5(
    output_file,
    raw_sourcefilename,
    raw,
    A_DATA,
    source_sink_results,
    source_sink_mean_results,
    patient_info,
):
    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, "w") as f:
        # write attributes
        f.attrs["description"] = (
            "HDF5 file containing iEEG data and calculations, A and \nsourcesink networks using code eeg_prep library provided \nfrom authors of 'Source-sink connectivity: a novel interictal EEG marker for seizure localization'\n\nThe data is structured as follows:\nmetadata: Contains metadata about the file\ndata: Contains windowed data, calculated A matrices and means, and reconstruction error\nnetwork: Contains sourcesink calculations from paper for the network calculations\n"
        )
        f.attrs["date_created"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        # ======== Metadata Group ========
        metadata_grp = f.require_group("metadata")

        metadata_grp.create_dataset(
            "sourcefilename", data=np.array(raw_sourcefilename, dtype="S")
        )

        channels = raw.ch_names
        metadata_grp.create_dataset(
            "channels", data=np.array(channels, dtype="S"), compression="gzip"
        )

        # patient info
        # reorder the "electrode" column to match the order of channels
        patient_info = patient_info.set_index("electrode")
        # Reorder the DataFrame based on the order of channels
        patient_info = patient_info.loc[channels].reset_index()

        patient_info_grp = metadata_grp.require_group("patient_info")

        # Save each column of the patient_info DataFrame as a dataset
        for col in patient_info.columns:
            # Convert the data to an appropriate format and save
            data = patient_info[col].values
            if patient_info[col].dtype == object:
                data = np.array(data, dtype="S")  # Convert strings to byte strings
            patient_info_grp.create_dataset(col, data=data)

        metadata_grp.create_dataset("sfreq", data=raw.info["sfreq"])

        winsize_samps = int(np.round(WINSIZE_MSEC * (sfreq / 1000)))
        stepsize_samps = int(np.round(STEPSIZE_MSEC * (sfreq / 1000)))
        # round to nearest sample
        metadata_grp.create_dataset("WINSIZE_MSEC", data=WINSIZE_MSEC)
        metadata_grp.create_dataset("WINSIZE_SAMPS", data=winsize_samps)

        metadata_grp.create_dataset("STEPSIZE_MSEC", data=STEPSIZE_MSEC)
        metadata_grp.create_dataset("STEPSIZE_SAMPS", data=stepsize_samps)

        n_samples = raw.n_times
        metadata_grp.create_dataset("N_samples", data=n_samples)

        A_mats, A_medians, A_thresh, A_mask, A_mean, window_indexs, windows, errors = (
            A_DATA
        )
        metadata_grp.create_dataset("N_wins", data=len(window_indexs))

        # ======== Data Group ========
        data_grp = f.require_group("data")
        data_grp.attrs["description"] = (
            "windows are shaped (winsize_samps, len(channels), n_wins)\nA_mats are shaped (len(channels), len(channels), n_wins)\nA_mean is shaped (len(channels), len(channels), 1) and is calcuated without outliers (A_mask==1)"
        )
        # raw data
        data_grp.create_dataset("window_indexs", data=window_indexs, compression="gzip")
        data_grp.create_dataset("windows", data=windows, compression="gzip")

        # reconstruction error
        data_grp.create_dataset(
            "window_reconstructionError", data=errors, compression="gzip"
        )

        # A data
        data_grp.create_dataset("A_mats", data=A_mats, compression="gzip")
        data_grp.create_dataset("A_medians", data=A_medians, compression="gzip")
        data_grp.create_dataset("A_thresh", data=A_thresh)
        data_grp.create_dataset("A_mask", data=A_mask, compression="gzip")
        data_grp.create_dataset("A_mean", data=A_mean, compression="gzip")

        # ======== Network Calculations Group ========
        network_grp = f.require_group("network")
        network_grp.attrs["description"] = (
            "contains the calculations of the sourcesink network\nfor corresponding the raw data and the A_mean data.\nVarients of calculations exist for including/excluding self-connections on the ranking stage (rank) and 'influence'/'conn' calculations (conn).\n\nThe data is structured as follows:\nraw: Contains the raw data calculations\nmean: Contains the A_mean data calculations\n"
        )

        raw_grp = network_grp.require_group("raw_sinksource")
        mean_group = network_grp.require_group("mean_sinksource")

        ss_ind_mats = source_sink_results["ss_ind_mats"]
        source_dist_mats = source_sink_results["source_dist_mats"]
        sink_dist_mats = source_sink_results["sink_dist_mats"]
        source_infl_mats = source_sink_results["source_infl_mats"]
        sink_conn_mats = source_sink_results["sink_conn_mats"]

        raw_grp.create_dataset(
            "ss_ind_mats", data=ss_ind_mats, compression="gzip"
        )
        raw_grp.create_dataset(
            "source_dist_mats", data=source_dist_mats, compression="gzip"
        )
        raw_grp.create_dataset(
            "sink_dist_mats", data=sink_dist_mats, compression="gzip"
        )
        raw_grp.create_dataset(
            "source_infl_mats", data=source_infl_mats, compression="gzip"
        )
        raw_grp.create_dataset(
            "sink_conn_mats", data=sink_conn_mats, compression="gzip"
        )

        # same for A_mean, drop "mean" from variables since it has the same structure
        mean_sink_conn_mats = source_sink_mean_results["sink_conn_mats"]
        mean_source_infl_mats = source_sink_mean_results["source_infl_mats"]
        mean_sink_dist_mats = source_sink_mean_results["sink_dist_mats"]
        mean_source_dist_mats = source_sink_mean_results["source_dist_mats"]
        mean_ss_ind_mats = source_sink_mean_results["ss_ind_mats"]

        mean_group.create_dataset(
            "ss_ind_mats", data=mean_ss_ind_mats, compression="gzip"
        )
        mean_group.create_dataset(
            "source_dist_mats", data=mean_source_dist_mats, compression="gzip"
        )
        mean_group.create_dataset(
            "sink_dist_mats", data=mean_sink_dist_mats, compression="gzip"
        )
        mean_group.create_dataset(
            "source_infl_mats", data=mean_source_infl_mats, compression="gzip"
        )
        mean_group.create_dataset(
            "sink_conn_mats", data=mean_sink_conn_mats, compression="gzip"
        )


# MAIN

# scan for all edf files in the directory
# check os windows or linux
if os.name == "nt":
    edf_path = "F:\\data\\iEEG\\raw_ieeg\\baseline_patients\\baseline_edfs"
    output_path = "F:\\git\\network_miner\\temp\\data"
    mapping_path = "f:\\manuscripts\\manuiscript_0001_hfo_rates\\data\\FULL_composite_patient_info.csv"
    ilae_path = "c:\\Users\\wirel\\Downloads\\ravi_hfo_numbers~N59+v03.csv"
    bad_channels_path = (
        "F:\\manuscripts\\manuiscript_0001_hfo_rates\\data\\bad_ch_review.xlsx"
    )
else:
    edf_path = "/media/dan/Data/data/iEEG/raw_ieeg/baseline_patients/baseline_edfs"
    output_path = "processed_files"
    mapping_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/data/FULL_composite_patient_info.csv"  
    ilae_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/ravi_hfo_numbers~N59+v03.csv"
    bad_channels_path = "/media/dan/Data/manuscripts/manuiscript_0001_hfo_rates/data/bad_ch_review.xlsx"
edf_files = sorted(list(glob.glob(os.path.join(f"{edf_path}", "*.EDF"))))

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


for file in tqdm(edf_files, desc="Processing files"):
    # get pid
    pid = int(os.path.basename(file).split("_")[0])
    if pid != 64:
        print(f"skipping {pid} because it is not 64")
        continue
    # get patient info
    patient_info = mappings[mappings["pid"] == pid]

    # ======== Process A and sourcesink ========
    # load the raw data
    raw = read_edf(file, preload=True)
    raw.info["line_freq"] = 60

    raw = preprocess_ieeg(raw)

    # remove bad channels
    raw = raw.drop_channels(patient_info[patient_info["bad_channel"] == 1]["electrode"].values)

    # average reference
    raw = raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

    sfreq = raw.info["sfreq"]

    # round to nearest sample
    winsize_samps = int(np.round(WINSIZE_MSEC * (sfreq / 1000)))
    stepsize_samps = int(np.round(STEPSIZE_MSEC * (sfreq / 1000)))

    # compute A
    calculations = compute_A(raw, winsize_samps, stepsize_samps)
    A_mats = calculations[0]
    A_mean = calculations[4]

    # compute sourcesink 
    ss_ind_mats,source_dist_mats,sink_dist_mats,source_infl_mats,sink_conn_mats,_, = state_sourcesink_array(A_mats,return_all=True,progressbar=True)
    source_sink_results = {
        "ss_ind_mats": ss_ind_mats,
        "source_dist_mats": source_dist_mats,
        "sink_dist_mats": sink_dist_mats,
        "source_infl_mats": source_infl_mats,
        "sink_conn_mats": sink_conn_mats,
    }

    source_sink_mean_results = {}
    mss_ind_mats,msource_dist_mats,msink_dist_mats,msource_infl_mats,msink_conn_mats,_, = state_sourcesink_array(A_mean,return_all=True,progressbar=True)
    source_sink_mean_results = {
        "ss_ind_mats": mss_ind_mats,
        "source_dist_mats": msource_dist_mats,
        "sink_dist_mats": msink_dist_mats,
        "source_infl_mats": msource_infl_mats,
        "sink_conn_mats": msink_conn_mats,
    }

    # # save to h5
    output_file = os.path.join(
        output_path, f"{pid:03}_{WINSIZE_MSEC:06}_{STEPSIZE_MSEC:06}.hdf5"
    )

    save_h5(
        output_file,
        file,
        raw,
        calculations,
        source_sink_results,
        source_sink_mean_results,
        patient_info,
    )
