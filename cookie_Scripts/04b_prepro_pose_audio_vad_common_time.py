import numpy as np
import pickle
import pandas as pd
import os


def time_match(time_base, time_new, dt):
     
    t0 = np.argmin(np.abs(time_new - time_base[0]))
    offset = time_base[0] - time_new[t0]
    assert( dt >= np.abs(offset))
    time_new_of = time_new + offset
    tend = np.argmin(np.abs(time_new_of - time_base[-1]))
    
    return time_new_of, t0, tend, offset

# Get dataset
# fname = 'Oculomotor_Cookie_Theft_face_dataset_cut_media_dwnsmp-4.8_flhNone-30_min_win4_audio.pickle'
fname = 'Oculomotor_Cookie_Theft_face_deID_dataset_cut_media_dwnsmp-4.8_flhNone-30_min_win4_audio.pickle'
with open(fname, "rb") as f:
    dataset = pickle.load(f)


dataset_resample = {}
for k in list(dataset.keys()):
    
    dataset_resample[k]= {}
    dataset_resample[k]['S_dB'] = []
    dataset_resample[k]['S_dB_times'] = [] 
    dataset_resample[k]['vad_bool'] = []
    dataset_resample[k]['vad_times'] = []
    
    dataset_resample[k]['beh'] =  dataset[k]['beh']
    dataset_resample[k]['ts'] =  []
    dataset_resample[k]['timestamp'] =  []
    
    fs = dataset[k]['ts_Fs']
    ts = dataset[k]['ts']
    ts_time = dataset[k]['frame_time']
    
    dt = 1/fs
    
    print(k)

    for ts_win, time_win in zip(ts, ts_time):
        
        time_aud,t0,tend, offset1 = time_match(time_win, dataset[k]['S_dB_times'], dt)
        
        win_aud = dataset[k]['S_dB'][t0:tend+1]
        time_aud = time_aud[t0:tend+1]        
        
        time_vad,t0,tend, offset2 = time_match(time_win, dataset[k]['vad_times'], dt)
        
        fs_vad = round(np.mean(1/np.diff(dataset[k]['vad_times'])))
        ratio = int(fs_vad/fs)
        
        win_vad = dataset[k]['vad_bool'][t0:tend+1:ratio]
        time_vad = time_vad[t0:tend+1:ratio]   
        
        if ts_win.shape[0] != win_vad.shape[0] or time_win.shape[0] != time_aud.shape[0]:
            tend = min(time_win.shape[0], win_vad.shape[0], time_aud.shape[0])
            ts_win = ts_win[:tend]
            time_win = time_win[:tend]
            win_aud = win_aud[:tend]
            time_aud = time_aud[:tend]
            win_vad = win_vad[:tend]
            time_vad = time_vad[:tend]
            
        dataset_resample[k]['ts'].append(ts_win)
        dataset_resample[k]['timestamp'].append(time_win)
        dataset_resample[k]['S_dB'].append(win_aud)
        dataset_resample[k]['S_dB_times'].append(time_aud)
        dataset_resample[k]['vad_bool'].append(win_vad)
        dataset_resample[k]['vad_times'].append(time_vad)
        # print('\t', offset1, offset2, time_win.shape[0], win_vad.shape[0], time_aud.shape[0])
        
        if ts_win.shape[0] != win_vad.shape[0] or time_win.shape[0] != time_aud.shape[0]:
            print('\t', ts_win.shape[0], win_vad.shape[0], time_aud.shape[0])

        
# fname = f'Oculomotor_Cookie_Theft_face_dataset_cut_media_flhNone-30_min_win4_audio_commFs{int(fs)}.pickle'
fname = f'Oculomotor_Cookie_Theft_face_deID_dataset_cut_media_flhNone-30_min_win4_audio_commFs{int(fs)}.pickle'
with open(fname, "wb") as f:
    pickle.dump(dataset_resample, f)

dir_name = fname.split('.pickle')[0]
os.makedirs(dir_name, exist_ok=True)

pose_cols = [f"{ax}_{n}" for n in range(478) for ax in ['x', 'y'] ]
aud_cols = [f"freq_{n}" for n in range(128)]
cols = ['timestamp'] + pose_cols + aud_cols + ['vad']

for k, v in dataset_resample.items():
    
    for n in range(len(v['ts'])):
        fcvsname = f"{k}-win_{n}.csv"
        
        tstmp = v['timestamp'][n].reshape((-1,1))
        ts = v['ts'][n]       
        ts = ts.reshape(ts.shape[0], -1)
        aud = v['S_dB'][n]
        vad = v['vad_bool'][n].reshape((-1,1))

        dat = np.hstack((tstmp, ts, aud, vad))
        
        df = pd.DataFrame(dat, columns=cols)
        df.to_csv(f"{dir_name}/{fcvsname}", index=False)

print(tstmp.shape, ts.shape, aud.shape, vad.shape)

# import matplotlib.pyplot as plt

# plt.scatter(time_vad, [1]*time_vad.size, marker='x')
# plt.show(block=False)

# plt.scatter(time_win, [1]*time_win.size, marker='o')
# plt.scatter(time_win, [1]*time_win.size, marker='o')

# plt.figure()
# plt.plot(time_aud - time_win)
# plt.show(block=False)

# plt.figure()
# plt.plot(np.diff(time_vad))

# plt.figure()
# plt.plot(time_vad)
# plt.show(block=False)
