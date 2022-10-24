import os

import mne
import numpy as np

if __name__ == '__main__':
    # change the path to the dataset folder
    abnormal = "./data/00000016_s004_t000.edf"
    normal = "./data/00000021_s004_t000.edf"

    ab_raw = mne.io.read_raw(abnormal, preload=True)
    norm_raw = mne.io.read_raw(abnormal, preload=True)

    ab_raw.filter(1., 40., fir_design='firwin', n_jobs=1)
    norm_raw.filter(1., 40., fir_design='firwin', n_jobs=1)

    fig = ab_raw.plot(n_channels=30)

    # data shape
    print(ab_raw._data.shape, norm_raw._data.shape)

    # min and max value
    np.min(ab_raw._data), np.max(ab_raw._data)

    chnames = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
               'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz',
               'Cz', 'Pz', ]

    short_ch_names = sorted([
        'A1', 'A2',
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'])
    ar_ch_names = sorted([
        'EEG A1-REF', 'EEG A2-REF',
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
    le_ch_names = sorted([
        'EEG A1-LE', 'EEG A2-LE',
        'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
        'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
        'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
        'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'])
    assert len(short_ch_names) == len(ar_ch_names) == len(le_ch_names)
    ar_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
        ar_ch_names, short_ch_names)}
    le_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
        le_ch_names, short_ch_names)}
    ch_mapping = {'ar': ar_ch_mapping, 'le': le_ch_mapping}

    reference = ab_raw.ch_names[0].split('-')[-1].lower()
    assert reference in ['le', 'ref'], 'unexpected referencing'
    reference = 'le' if reference == 'le' else 'ar'
    ab_raw.rename_channels(ch_mapping[reference])
    norm_raw.rename_channels(ch_mapping[reference])

    ab_raw = ab_raw.pick_channels(chnames, ordered=True)
    norm_raw = ab_raw.pick_channels(chnames, ordered=True)

    montage = mne.channels.make_standard_montage("standard_1020")

    ab_raw.set_montage(montage)
    norm_raw.set_montage(montage)

    ab_raw.resample(sfreq=100)
    norm_raw.resample(sfreq=100)

    ab_csd = mne.preprocessing.compute_current_source_density(ab_raw)

    norm_csd = mne.preprocessing.compute_current_source_density(norm_raw)

    x = ab_csd.get_data()
    xnr = norm_csd.get_data()


    def multichannel_sliding_window(X, size, step):
        """
        Generate sliding windows from multichannel 1d data and outputs
        X = np.arange(50).reshape((5, 10))
        print(X)
        [[ 0  1  2  3  4  5  6  7  8  9]
         [10 11 12 13 14 15 16 17 18 19]
         [20 21 22 23 24 25 26 27 28 29]
         [30 31 32 33 34 35 36 37 38 39]
         [40 41 42 43 44 45 46 47 48 49]]
        print(multichannel_sliding_window(X, 4, 2))
        [[[ 0  1  2  3]
          [10 11 12 13]
          [20 21 22 23]
          [30 31 32 33]
          [40 41 42 43]]
         [[ 2  3  4  5]
          [12 13 14 15]
          [22 23 24 25]
          [32 33 34 35]
          [42 43 44 45]]
         [[ 4  5  6  7]
          [14 15 16 17]
          [24 25 26 27]
          [34 35 36 37]
          [44 45 46 47]]]
        """
        shape = (X.shape[0] - X.shape[0] + 1, (X.shape[1] - size + 1) // step, X.shape[0], size)
        strides = (X.strides[0], X.strides[1] * step, X.strides[0], X.strides[1])
        return np.lib.stride_tricks.as_strided(X, shape, strides)[0]


    b = multichannel_sliding_window(x, 200, 100)
    c = multichannel_sliding_window(xnr, 200, 100)

    print(b.shape, c.shape)

    # ab=np.expand_dims( b,axis=1)
    # nr=np.expand_dims( c,axis=1)
    ab = np.expand_dims(b, axis=1)
    nr = np.expand_dims(c, axis=1)

    print(ab.shape, nr.shape)

    # ! rm - r "/content/train"

    os.makedirs("./train/abnormal", exist_ok=True)
    os.makedirs("./train/normal", exist_ok=True)
    os.makedirs("./val/abnormal", exist_ok=True)
    os.makedirs("./val/normal", exist_ok=True)

    for i in range(4):
        np.save(f"./train/abnormal/abnorm_{i}", ab, allow_pickle=True)
        np.save(f"./train/normal/norm_{i}", nr, allow_pickle=True)

    # TODO make not same as train
    for i in range(4):
        np.save(f"./val/abnormal/abnorm_{i}", ab, allow_pickle=True)
        np.save(f"./val/normal/norm_{i}", nr, allow_pickle=True)
