

import pickle
from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
import torch

fname = 'Oculomotor_Cookie_Theft_face_dataset_cut_media_dwnsmp-4_flhNone-30_audio.pickle'

with open(fname, "rb") as f:
    dataset = pickle.load(f)

k = list(dataset)[1]


win_len = 2
hop = 1
vad_min = .5
ts_winds, audio_winds, inx_winds, y_class, sbj_id = [], [], [], [], []
nan_winds_g, silent_winds_g, smal_g = [], [], []

for k in dataset:
    
    frame_times = dataset[k]['frame_time']
    S_times = dataset[k]['S_dB_times']
    vad_times = dataset[k]['vad_times']
    fps = dataset[k]['ts_Fs']
    
    nan_winds, silent_winds = 0, 0
    smal,ts_wind_sj = [], []
    for t in np.arange(frame_times[0], frame_times[-1]+1/fps, hop):
        # VAD
        t_end = t + win_len
        vad_inx = (vad_times >= t) & (vad_times < t_end)
        vad_mean =  dataset[k]['vad_bool'][vad_inx].mean()
        
        if vad_mean < vad_min:  # ignore if less than vad_min
            silent_winds += 1
            continue
        
        # FACE
        ts_inx = (frame_times > t) & (frame_times <= t_end)
        ts_win = dataset[k]['ts'][:,:, ts_inx]
        
        if np.isnan(ts_win.sum(0).sum(0)).any(): # ignore if NANs or empy
            nan_winds += 1
            continue
        elif ts_win.sum() == 0:
            continue
        elif ts_win.shape[-1] < win_len*fps: # ignore if small win
            smal.append(ts_win.shape[-1])
            continue
        ts_win = ts_win.reshape([-1, ts_win.shape[-1]])
        
        # AUDIO
        S_ix = (S_times > t) & (S_times <= t_end)
        S_win = dataset[k]['S_dB'][S_ix]
        
        ts_wind_sj.append(ts_win)
        audio_winds.append(S_win)
        inx_winds.append(t)
        
    ts_winds.extend(ts_wind_sj) 
    
    nan_winds_g.append(nan_winds)
    silent_winds_g.append(silent_winds)
    smal_g.append(smal)
        
    y_class.extend([dataset[k]['beh']['gen_diagnosis']]* len(ts_winds))
    sbj_id.extend([k]* len(ts_wind_sj))
    

