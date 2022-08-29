#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:13:24 2021

@author: adonay
"""

from glob import glob
import os.path as op

import json
import numpy as np
import pickle
import pandas as pd
import cv2
import mediapipe as mp
import time

NUM_FACE = 1

class FaceLandMarks():
    def __init__(self, staticMode=False,maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, max_num_faces=self.maxFace,
                                                 refine_landmarks=True,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces



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

out_path = '/data/Dropbox (Partners HealthCare)/Oculomotor_Cookie_Theft/*.MOV'
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


dataset = {}
#[ i for i, v in enumerate(vids) if v == vid]
for vid in vids[1:]:
    t0 = time.time()
    base_name = op.basename(vid)[:-4]
    sbj_id = base_name[:16]
    dir_name = op.dirname(vid)
    
    if sbj_id in list(out_files):
        print(f"Skipping {sbj_id}")
        continue
    
    cap = cv2.VideoCapture(vid)

    pTime = 0
    detector = FaceLandMarks()
    inx = 0 
    frame_inx = []
    ts = []
    success = True
    while success:
        success, img = cap.read()
        if not success:
            break
        img, faces = detector.findFaceLandmark(img, draw=False)
        if len(faces)!=0:
            face = faces[0]
            face = np.stack(face)
        else:
            face = np.zeros((478,2))
            print(f"empty {inx} in {sbj_id}")
        
        ts.append(face)
        frame_inx.append(inx)
        inx += 1
        
    assert(len(frame_inx))
    
    tts = np.stack(ts)
    if sbj_id == "10348_2020_02_14": # BEH date differ
        dataset[sbj_id] = {"ts": tts, 
                           "frame_ix": frame_inx, 
                           "beh": df_beh.loc['10348_2020_02_19']
                           }
        print(f"The {sbj_id} subj with diff beh date: 10348_2020_02_19")
        continue
    
    dataset[sbj_id] = {"ts": tts,
                       "frame_ix": frame_inx,
                       "beh": df_beh.loc[sbj_id]}
    print(f"Done {sbj_id} in {(time.time() - t0)/60:.2f}")



with open("Oculomotor_Cookie_Theft_face_dataset_inx_mediapipe.pickle", "wb") as f:
      pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1,1, sharex=True)
# plt.gca().invert_yaxis()

# axs.scatter(face[:,0], face[:,1])

# axs.set_xlabel