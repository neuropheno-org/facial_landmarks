#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:02:07 2021

@author: adonay
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from mne.filter import filter_data, resample

def wind_nonans(out, frame_inx, min_samp_win):
    # out shape = [samples, loc, ax]
    #returns list windowed data without nans and new frame inx
    
    no_nans = ~np.isnan(out[:,0,0])
    
    indices = np.nonzero(no_nans[1:] != no_nans[:-1])[0] + 1
    b = np.split(out, indices)
    b = b[0::2] if no_nans[0] else b[1::2]
    f_inx = np.split(frame_inx, indices)
    f_inx = f_inx[0::2] if no_nans[0] else f_inx[1::2]
    
    b = [bb for bb in b if bb.shape[0] >= min_samp_win]
    f_inx = [bb for bb in f_inx if bb.shape[0] >= min_samp_win]

    return b, f_inx


out_files = {"10072_2018_06_27" : "face cropped",
             "10079_2018_05_30": "not doing task",
             "10029_2018_08_01": "not much talking",
             '10049_2019_12_23' : "face cropped",
             "10120_2019_01_11": "Doesn't talk much",
             "10120_2019_10_12": "no talking",
             "10120_2020_01_10": "no talking",
             "10122_2019_01_11": "little girl just words",
             "10130_2019_01_12": "not much talking",
             "10130_2020_01_11": "not much talking",
             "10207_2018_07_12": "face cropped",
             "10208_2018_07_16" : "child moving all over, occluding with hands",
             "10209_2018_07_16": "child not talking",
             "10272_2019_01_10": "not looking",
             "10275_2019_01_11": "kid moving",
             "10276_2019_01_11" : "child moving all over, occluding with hands",
             "10282_2019_01_12":"no talking",
             "10323_2019_11_18":"no talking",
             }

# "Oculomotor_Cookie_Theft_face_deID_dataset_pose_audio.pickle"
with open("Oculomotor_Cookie_Theft_face_dataset_pose_audio.pickle", "rb") as f:
    dataset = pickle.load(f)
# "Oculomotor_Cookie_Theft_face_deID_dataset_pose_audio.pickle"
with open("Oculomotor_Cookie_Theft_face_dataset_pose_audio.pickle", "rb") as f:
    dataset_audio = pickle.load(f)

times0 = pd.read_csv('/data/databases/Ataxia_dataset/pose_analysis/cookie_videos_tstart.csv', index_col=0)

path_times = "/data/Dropbox (Partners HealthCare)/Data/Subjects_Activity/"

out_list = list(out_files.keys())

lfq = None
hfq = 30
dsmpl_fact = 4.8  
min_win = 4  # seconds





for k in list(dataset.keys()):
    if k in out_list:
        print("should be out:", k)
        del dataset[k]
        continue


    dataset[k]["ts_Fs"] = dataset_audio[k]["ts_Fs"]
    dataset[k]["beh"] = dataset_audio[k]["beh"]
    dataset[k]['S_dB_Fs'] = dataset_audio[k]['S_dB_Fs']
    dataset[k]['S_dB_duration'] = dataset_audio[k]['S_dB_duration']
    dataset[k]['S_dB'] = dataset_audio[k]['S_dB']
    dataset[k]['S_dB_times'] = dataset_audio[k]['S_dB_times']
    dataset[k]['vad_bool'] = dataset_audio[k]['vad_bool']
    dataset[k]['vad_times'] = dataset_audio[k]['vad_times'] 
    dataset[k]['vad_times_frame_duration'] = dataset_audio[k]['vad_times_frame_duration']
    
    try:
        t0 = times0.loc[k+"_cookie_theft.MOV", "pk"]
        t0 = t0*240//44100 # 240 hz video, 44000 hz audio
    except KeyError:
        print("missing ", k)
        t0 = 240 # cut a sec

    if k == "10348_2020_02_14":
        times_file = path_times + "10348_2020_02_19_ipad_ts.xlsx"
    else:
        times_file = path_times + k + "_ipad_ts.xlsx"

    tinfo = pd.read_excel(times_file)

    try:
        mask = tinfo.loc[ tinfo["TaskName"]=="Cookie_Theft"]
    except:
        mask = tinfo.loc[ tinfo["Task Name"]=="Cookie_Theft"]

    ts = dataset[k]['ts']
    fs = dataset[k]["ts_Fs"]
    
    if len(mask) == 0:
        print("no time info ", k)
        tlen_samp = len(ts) - t0
    else:
        if len(mask) > 1:  # First try date
            date = k[6:].replace("_", "-")
            try:
                mask = mask.loc[mask['Date (EST)'] == date]
            except KeyError:
                mask = mask.loc[mask['Date_EST_'] == date]
                
            if len(mask) > 1:  # then use last
                mask = mask.iloc[len(mask)-1]

        try:
            tleng = mask["Stage2VideoEndTime_UNIX_"] - mask["Stage2VideoStartTime_UNIX_"]
        except:
            tleng = mask['Stage 2 Video End time (UNIX)'] - mask['Stage 2 Video Start time (UNIX)']
            tleng = tleng/1000  # in secs
        
        try:
            tlen_samp = tleng.values[0] * fs
        except:
            tlen_samp = tleng * fs

    if tlen_samp + t0 > len(ts):
        print(f"Duration: {dataset[k]['S_dB_duration']}, len ts {len(ts)/fs:.2f}, len sess {tlen_samp/fs:.2f} | {t0/fs:.2f} : {(t0+tlen_samp)/fs:.2f}")
        print(len(ts), tlen_samp-t0, t0, tlen_samp)

    tend = min(len(ts), tlen_samp+t0)
    ts = ts[int(t0):int(tend),:,:]
    dataset[k]['frame_ix'] = dataset[k]['frame_ix'][int(t0):int(tend)]

    
    ts = ts.astype(float)
    # nan if mean x coord is <0
    ts[ts[:,:,0].mean(1) <= 0] = np.nan
    out = np.transpose(filter_data(np.transpose(ts, (1,2,0)), fs, lfq, hfq) , (2,0,1))
    f_inx = np.array(dataset[k]['frame_ix'], dtype=float)

    min_samp_win = min_win * fs
    out_win, f_inx = wind_nonans(out, f_inx, min_samp_win)
    
    f_time = [f_x * (1/fs) for f_x in f_inx]
    
    ts_out = [resample(win,down=dsmpl_fact, npad='auto', axis=0) for win in out_win]
    
    new_fs = dataset_audio[k]["ts_Fs"]/dsmpl_fact
    f_time = [np.arange(t[0],t[-1],1/new_fs)[:-1] for t in f_time]

    dataset[k]['ts'] = ts_out
    dataset[k]['frame_ix'] = None
    dataset[k]['frame_time'] = f_time
    dataset[k]["ts_Fs"] = new_fs
   
# "Oculomotor_Cookie_Theft_face_deID_dataset_cut_media_dwnsmp-{dsmpl_fact}_flh{lfq}-{hfq}_min_win{min_win}_audio.pickle" 
with open(f"Oculomotor_Cookie_Theft_face_dataset_cut_media_dwnsmp-{dsmpl_fact}_flh{lfq}-{hfq}_min_win{min_win}_audio.pickle", "wb") as f:
    pickle.dump(dataset, f)

