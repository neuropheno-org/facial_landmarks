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
from u_filtering import filter_predictions

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

with open("Oculomotor_Cookie_Theft_face_dataset_pose_audio.pickle", "rb") as f:
     dataset = pickle.load(f)


times0 = pd.read_csv('/data/databases/Ataxia_dataset/pose_analysis/cookie_videos_tstart.csv', index_col=0)

path_times = "/data/Dropbox (Partners HealthCare)/Data/Subjects_Activity/"

filtertype = "median" #"arima" #"median"#
downsample_fact = 2
out_list = list(out_files.keys())

for k in list(dataset.keys()):
    if k in out_list:
        print("should be out:", k)
        continue

    try:
        t0 = times0.loc[k+"_cookie_theft.MOV", "pk"]
        t0 = t0*240//44100 # 240 hz video, 44000 hz audio
    except KeyError:
        print("missing ", k)
        continue

    if k == "10348_2020_02_14":
        times_file = path_times + "10348_2020_02_19_ipad_ts.xlsx"
    else:
        times_file = path_times + k + "_ipad_ts.xlsx"

    tinfo = pd.read_excel(times_file)

    try:
        mask = tinfo.loc[ tinfo["TaskName"]=="Cookie_Theft"]
    except:
        mask = tinfo.loc[ tinfo["Task Name"]=="Cookie_Theft"]

    if len(mask) == 0:
        print("no time info ", k)
        continue
    elif len(mask) > 1:
        mask = mask.iloc[len(mask)-1]


    try:
        tleng = mask["Stage2VideoEndTime_UNIX_"] - mask["Stage2VideoStartTime_UNIX_"]
    except:
        tleng = mask['Stage 2 Video End time (UNIX)'] - mask['Stage 2 Video Start time (UNIX)']

    tleng = tleng/1000  # in secs
    try:
        tlen_samp = tleng.values[0] *240
    except:
        tlen_samp = tleng *240

    ts = dataset[k]['ts']

    if tlen_samp-t0 > len(ts):
        print(len(ts), tlen_samp-t0, t0, tlen_samp)

    tend = min(len(ts), tlen_samp, tlen_samp+t0)
    ts = ts[int(t0):int(tend),:,:]
    ts1 = filter_predictions(ts,
                          filtertype=filtertype,
                          windowlength=5,
                          p_bound=0.01)


    # ts = filter_predictions(ts, "median",windowlength=downsample_fact)
    ts = ts[::downsample_fact]

    if filtertype == "arima":
        ts = filter_predictions(ts, "spline")

    dataset[k]['ts'] = ts

with open(f"Oculomotor_Cookie_Theft_face_dataset_cut_filt-{filtertype}_dwnsmp-{downsample_fact}.pickle", "wb") as f:
     pickle.dump(dataset, f)



if 0:
    ts = dataset['10268_2018_12_20']['ts']
    ts = dataset['10056_2018_04_11']['ts']

    ts_f = filter_predictions(ts,
                              filtertype="median",
                              windowlength=5,
                              p_bound=0.001)

    inxs = [584, 1775]#[4584, 6775]
    fig, axs = plt.subplots(3,1)
    inx = np.arange(inxs[0], inxs[1])
    axs[0].plot(inx, ts[inx,:,0])
    axs[1].plot(inx,ts_f[inx,:, 0])

    ts_f2 = ts_f[::3]
    ts_f2 = filter_predictions(ts_f2, "median", windowlength=5)
    ds = 3
    inxds = np.arange(round(inxs[0]/3), round(inxs[1]/3))
    axs[2].plot(inx[::ds], ts_f[inxds, :, 0])


    fig, axs = plt.subplots(1,3, sharex=True)
    plt.gca().invert_yaxis()
    for i, inx in enumerate([10, 600, 900]):
        axs[i].scatter(ts[inx,:, 0], ts[inx,:, 1], c=ts[inx,:, 2])
        # axs[i].set_xlabel

    plt.figure()
    inx = 4387
    plt.scatter(ts[inx,:, 0], ts[inx,:, 1], c=ts[inx,:, 2])
    plt.gca().invert_yaxis()
