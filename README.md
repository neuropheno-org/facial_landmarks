# facial_landmarks


## Cookie theft task
Classify Ataxia, PD and controls from facial movements during cookie theft. Lasts between 20 sec to 1 min. For prediction, a moving window of 100 samples with 50% overlap is used to classify between two groups. 

`01_prepro_vids_cookies_mediapipe.py`
Facial landmarks were extracted using mediapipe.
Output file: Oculomotor_Cookie_Theft_face_dataset_inx_mediapipe.pickle

`02_video_get_beep_pk.py`
Detect beep indicating start of the task
Output file: cookie_videos_tstart.csv

`03_get_audio_n_speech.py`
Gets the audio of the videos and using Voice Activity Detector marks if there is speech at each window of 10 ms. 
Output file: Oculomotor_Cookie_Theft_face_dataset_pose_audio.pickle"

`04_prepro_vids_cookies_filtering_mediapipe.py`
Filters frame data, downsample it, outputs dict of subj with dict with time series landmarks, audio, vad, behavioral variables
Output file: Oculomotor_Cookie_Theft_face_dataset_cut_media_dwnsmp-{dsmpl_fact}_flh{lfq}-{hfq}_audio.pickle

`04b_prepro_pose_audio_vad_common_time.py`
Get landmarks, audio and VAD in a common time and in one csv
Output file:  f'Oculomotor_Cookie_Theft_face_deID_dataset_cut_media_flhNone-30_min_win4_audio_commFs{int(fs)}.pickle'

`05_DL.py`
Does deep learning, saves a result figure plotting loss, f1, and accuracy. The outputs for the DL model can be all face points, mouse only, reduced to 74 points, or eye and mouse points. There are several models, inception short, inception, CNN (3d inputs), FCNN







## Go La Me tasks

Classify Ataxia, PD and controls from facial movements during the tasks. Each task lasts about 10 seconds. For prediction, a moving window of 100 samples with 50% overlap is used to classify between two groups. 

`01_prepro_vids_cookies_mediapipe.py`
Loops over a directory, reads the video, and estimates for each frame 480 facial landmarks using mediapipe. 
Output file: golame_face_dataset.pickle

`01a_face_align_golame.py`
Face points are aligned, first the eyes are fixed to two coordinates (similarityTransformMat) then the facial movements are transferred (transferExpression).
Output file: golame_face_deID_dataset_inx_mediapipe.pickle


`05_DL_golame.py`
Deep learning used to classify subjects. Currently implemented models are inceptionTime, short inceptionTime and CNN short fc.

Output files: 
 f"checkpoints/{net_name}_{data_input}_{task}_{win_len}wz_{hop}st_{g1}vs{g0}_mask_y_len{mask_y_len}_mask_x_len{mask_x_len}_lr{lr}_deriv{derivatives}_pntsubset{pose_pnts_small}_optim{optim}_valBatch{batch_val}_{time_now}"

f"result_figs/{title}_{data_input}_{task}_{win_len}wz_{hop}st_{g1}vs{g0}_mask_y_len{mask_y_len}_mask_x_len{mask_x_len}_lr{lr}_deriv{derivatives}_pntsubset{pose_pnts_small}_optim{optim}_valBatch{batch_val}_{round(ttot/60)}mintime.png"

`05_DL_golame_taskpred.py`
Similar to 05_DL_golame.py but instead of predicting patients’ group, it predicts which task the windowed data belongs to.

`05_DL_golame_transferl.py`
Applies transfer learning from the task prediction model to predicting patients group

`06_DL_saliency.py`
Does integrated gradients to see the saliency, i.e., to see how much a landmark contributes to classification.

`06_DL_saliency_bytasks.py`
Similarly as in  06_DL_saliency.py but for visualizing contribution in classifying tasks. 
