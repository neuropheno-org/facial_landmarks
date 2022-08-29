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
import os.path as op
import librosa
import librosa.display
import struct
import webrtcvad
import noisereduce as nr


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


n_fft = 512 # time window size for FFT
n_mels = 128 # number of mel bands
hop_length = 160 # number samples hop; at 22050 Hz, 512 samples ~= 23ms

vad = webrtcvad.Vad(1)

vids = glob.glob('/data/Dropbox (Partners HealthCare)/Oculomotor_Cookie_Theft/*.MOV')
vids.sort()

# "Oculomotor_Cookie_Theft_face_deID_dataset_inx_mediapipe.pickle"
with open("Oculomotor_Cookie_Theft_face_dataset_inx_mediapipe.pickle", "rb") as f:
    dataset = pickle.load(f)


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

dataset_new = {}
# def peak_detect(vid):
for i, vid in enumerate(vids):
    t0 = time.time()
    # try:
    my_clip = mp.VideoFileClip(vid)
    # except OSError:
        # continue

    base_name = op.basename(vid)[:-4]
    sbj_id = base_name[:16]

    if sbj_id in list(out_files):
        continue

    audio = my_clip.audio.to_soundarray()

    fs = my_clip.audio.fps
    duration = my_clip.audio.duration
    nsamples = len(audio)

    if fs*duration != nsamples:
        print(f"sbj_id: nsamples {nsamples} but fs*duration {fs*duration}, diff{nsamples - fs*duration}")

    sample_rate = 8000
    ys_re = librosa.resample(audio.T, fs, sample_rate)

    ys_re_mono_ = librosa.to_mono(ys_re)
    ys_re_mono = np.round(ys_re_mono_ *10000)
    ys_re_mono = ys_re_mono.astype(np.int)
    ys_re_mono = [max(min(x, 32767), -32768) for x in ys_re_mono] # must fall in this range\

    raw_y2 = struct.pack("%dh" % len(ys_re_mono), *ys_re_mono) # package data for VAD

    frame_duration = 10  # ms

    frames = frame_generator(frame_duration, raw_y2, sample_rate)
    frames = list(frames)

    vad_bool = np.full(len(frames), np.nan)
    vad_times = np.full(len(frames), np.nan)
    vad_bool2 = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        vad_bool[i] = vad.is_speech(frame.bytes, sample_rate) # use two different VAD params set above
        vad_times[i] = frame.timestamp


    ys_re_mono_ = nr.reduce_noise(y=ys_re_mono_, sr=sample_rate)
    ys_re_mono_ = (ys_re_mono_ -ys_re_mono_.mean()/ys_re_mono_.std())
    S = librosa.feature.melspectrogram(ys_re_mono_, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, window='hann')

    S_dB = librosa.power_to_db(S, ref=3) # ref scaled to arbitrary number
    S_dB = S_dB.T
    start = (hop_length/sample_rate)/2
    S_dB_times = np.linspace(start, duration, len(S_dB))

    # plt.figure()
    # librosa.display.waveplot(ys_re_mono_, sr=sample_rate)
    # axes = plt.gca()
    # x_min, x_max = axes.get_xlim()
    # y_min, y_max = axes.get_ylim()
    # plt.plot(vad_times,vad_bool*y_max)
    # plt.show(block=False)

    # img = librosa.display.specshow(S_dB.T, x_axis='s',y_axis='mel', sr=sample_rate, hop_length=hop_length)
    # plt.plot(vad_times, vad_bool*1800)

    # plt.plot(S_dB_times,[1800]*len(S_dB), "*")
    # plt.xlim([0, 10])
    # plt.show(block=False)

    dataset_new[sbj_id] = {}
    dataset_new[sbj_id]["ts"] = dataset[sbj_id]["ts"]
    dataset_new[sbj_id]["frame_ix"] = dataset[sbj_id]["frame_ix"]
    dataset_new[sbj_id]["ts_Fs"] = my_clip.fps
    dataset_new[sbj_id]["beh"] = dataset[sbj_id]["beh"]
    dataset_new[sbj_id]['S_dB_Fs'] = sample_rate
    dataset_new[sbj_id]['S_dB_duration'] = duration
    dataset_new[sbj_id]['S_dB'] = S_dB
    dataset_new[sbj_id]['S_dB_times'] = S_dB_times
    dataset_new[sbj_id]['vad_bool'] = vad_bool
    dataset_new[sbj_id]['vad_times'] = vad_times
    dataset_new[sbj_id]['vad_times_frame_duration'] = frame_duration


dataset = dataset_new
# "Oculomotor_Cookie_Theft_face_deID_dataset_pose_audio.pickle"
with open("Oculomotor_Cookie_Theft_face_dataset_pose_audio.pickle", "wb") as f:
      pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
