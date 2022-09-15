import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
import glob
import scipy.io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
import torch
import mne
import matplotlib.pyplot as plt



def load_file_names():
    all_files = glob.glob('./data/*.edf')
    normal_subjects = [f for f in all_files if "h" in f.split("/")[-1]]
    abnormal_subjects = [f for f in all_files if "s" in f.split("/")[-1]]
    return normal_subjects, abnormal_subjects


def read_file(file_path):
    datax=mne.io.read_raw_edf(file_path,preload=True)
    datax.set_eeg_reference()
    datax.filter(l_freq=1,h_freq=45)
    epochs=mne.make_fixed_length_epochs(datax,duration=25,overlap=0)
    epochs=epochs.get_data()
    return epochs #trials,channel,length


def load_all_data():
    

normal = load_file_names()[0] 
a = load_data(normal[0])
print(a.shape)

