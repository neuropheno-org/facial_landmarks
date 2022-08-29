#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:13:24 2021

@author: adonay
"""

from glob import glob
import os.path as op
import os
import json
import numpy as np
import pickle
import pandas as pd

out_path = '/data/databases/Ataxia_dataset/Oculomotor_Cookie_Theft/*[!original]/*.mp4'
vids = glob(out_path)
vids.sort()

fname_beh = '/home/adonay/Desktop/projects/Ataxia/2020_05_21_All_Tasks-data2.csv'
df_beh = pd.read_csv(fname_beh, index_col=0)

subjs = []
for vid in vids:
    base_name = op.basename(vid)[:-4]
    sbj_id = base_name[:16]
    subjs.append(sbj_id)
s, c = np.unique(subjs, return_counts=True)
print( f"Subj repeated: {s[c>1]}")


vid = '/data/databases/Ataxia_dataset/Oculomotor_Cookie_Theft/10251_2018_10_18_cookie_theft/10251_2018_10_18_cookie_theft.mp4'
dataset = {}
#[ i for i, v in enumerate(vids) if v == vid]
for vid in vids:
    base_name = op.basename(vid)[:-4]
    sbj_id = base_name[:16]
    dir_name = op.dirname(vid)

    jsns = glob(f"{dir_name}/{base_name}*.json")
    jsns.sort()

    ts = []
    frame_inx = []
    n_empty = 0
    for inx, js in enumerate(jsns):
        with open(js, "r") as f:
            data = json.load(f)

        if len(data["people"]) == 0:
            kk = np.zeros((70,3))
            kk[:,:2] = np.nan
            ts.append(kk)
            n_empty += 1
            frame_inx.append(inx)
            continue

        # if more than 2 faces detected, select the largest face
        i = 0
        if len(data["people"])> 1:
            max_face = []
            for i in range(len(data["people"])):
                kk = np.array(data["people"][i]["face_keypoints_2d"])
                kk = kk.reshape([-1, 3])
                max_face.append(max(kk[:,0]) -min(kk[:,0]))
            i = np.argmax(max_face)

        kk = np.array(data["people"][i]["face_keypoints_2d"])
        kk = kk.reshape([-1, 3])
        ts.append(kk)
        frame_inx.append(inx)
    tts = np.stack(ts)
    
    assert(len(frame_inx) == len(jsns)) # check no missing frames
    
    if n_empty:
        print(f"{sbj_id} had {n_empty} frames with no points")

    if sbj_id == "10348_2020_02_14": # BEH date differ
        dataset[sbj_id] = {"ts": tts, "frame_ix": frame_inx, "beh": df_beh.loc['10348_2020_02_19']
                           }
        print(f"The {sbj_id} subj with diff beh date: 10348_2020_02_19")
        continue
    dataset[sbj_id] = {"ts": tts, "frame_ix": frame_inx, "beh": df_beh.loc[sbj_id]}



with open("Oculomotor_Cookie_Theft_face_dataset_inx.pickle", "wb") as f:
      pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1,1, sharex=True)
# plt.gca().invert_yaxis()

# for i in range(2):
#     kk = np.array(data["people"][i]["face_keypoints_2d"])
#     kk = kk.reshape([-1, 3])
#     ts.append(kk)

#     axs.scatter(kk[:,0], kk[:,1])

# axs.set_xlabel