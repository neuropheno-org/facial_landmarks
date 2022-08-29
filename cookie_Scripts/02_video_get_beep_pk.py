#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:22:54 2021

@author: adonay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:53:10 2021

@author: adonay
"""

import moviepy.editor as mp
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import signal
import pickle
import glob
import pandas as pd
from multiprocessing.pool import ThreadPool
import os.path as op

vids = glob.glob('/data/Dropbox (Partners HealthCare)/Oculomotor_Cookie_Theft/*.MOV')
vids.sort()

df = pd.DataFrame(columns=["name", "pk"])



# def peak_detect(vid):
for vid in vids:
    t0 = time.time()
    try:
        my_clip = mp.VideoFileClip(vid)
    except OSError:
        continue

    audio = my_clip.audio.to_soundarray()
    fs = my_clip.audio.fps
    audio = audio[:audio.shape[0]//2,:]
    # filter with n taps = fs, window width 256
    b = signal.firwin(fs, [490, 500], width=256, pass_zero=False, fs=fs)
    y2 = signal.convolve(audio[:,0],b , mode='same')

    # get envelope
    s_h = signal.hilbert(y2)
    s_en = np.abs(s_h)

    s_en = np.convolve(s_en, np.ones(fs)/fs, mode='same')

    b = signal.firwin(fs, [395, 405], width=256, pass_zero=False, fs=fs)
    y2 = signal.convolve(audio[:,0], b, mode='same')
    s_h = signal.hilbert(y2)
    s_en1 = np.abs(s_h)
    s_en1 = np.convolve(s_en1, np.ones(fs)/fs, mode='same')

    y = s_en/s_en1

    pk = np.argmax(y)

    name = vid.split('/')[-1]
    # fname = f"/data/databases/Ataxia_dataset/pose_analysis/peak_detection/{name}.p"
    # pickle.dump(pk, open(fname, "wb"))

    df = df.append({"name": name, "pk": pk} ,ignore_index=True)
    print(f"done {name}, t:{time.time()- t0}")

df = df.set_index("name")
# [peak_detect(vid) for vid in vids]
# pool = ThreadPool(processes=2)
# pool.map(peak_detect, (vid for vid in vids))
# pool.close()

time_manual = {
    "10252_2018_10_18_cookie_theft.mov": 11*44100,
    "10259_2018_11_14_cookie_theft.MOV": 0,
    "10324_2019_11_18_cookie_theft.MOV": 32*44100,
    "10352_2020_03_03_cookie_theft.MOV": 0,
    "10094_2019_03_13_cookie_theft.MOV": 11*44100,
    "10056_2018_04_11_cookie_theft.MOV": 0,
    "10079_2018_05_30_cookie_theft.MOV":0,
    "10192_2018_05_31_cookie_theft.MOV":0,
    "10255_2020_01_11_cookie_theft.MOV": 6*44100,
    "10257_2018_11_07_cookie_theft2.MOV": 0,
    "10285_2019_01_31_cookie_theft.MOV": 0,
    "10346_2020_02_05_cookie_theft.MOV": 2*44100
    }

df_manual = pd.DataFrame.from_dict(time_manual,orient="index", columns=["pk"])

df = df_manual.combine_first(df)




pname ="/data/databases/Ataxia_dataset/pose_analysis/"
fname = "cookie_videos_tstart"
inx = 1
while True:
    if not op.exists(pname + fname + ".csv"):
        df.to_csv(pname + fname + ".csv")
        break
    else:
        fname = fname[:20] +"_"+ str(inx)
        inx +=1

