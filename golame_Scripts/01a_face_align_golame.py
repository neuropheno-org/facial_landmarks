

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os 
import cv2, math
import numpy as np
import glob
import pandas as pd

def similarityTransformMat(initialPoints, destinationPoints):
        sin60 = math.sin(60*math.pi / 180)
        cos60 = math.cos(60*math.pi / 180)

        #third point is caluculated for initial points
        xin = cos60*(initialPoints[0][0] - initialPoints[1][0]) - sin60*(initialPoints[0][1] - initialPoints[1][1]) + initialPoints[1][0]
        yin = sin60*(initialPoints[0][0] - initialPoints[1][0]) + cos60*(initialPoints[0][1] - initialPoints[1][1]) + initialPoints[1][1]

        initialPoints = np.append(initialPoints,[ [int(xin), int(yin)]], axis=0)

        #third point is caluculated for destination points
        xout = cos60*(destinationPoints[0][0] - destinationPoints[1][0]) - sin60*(destinationPoints[0][1] - destinationPoints[1][1]) + destinationPoints[1][0]
        yout = sin60*(destinationPoints[0][0] - destinationPoints[1][0]) + cos60*(destinationPoints[0][1] - destinationPoints[1][1]) + destinationPoints[1][1]

        destinationPoints= np.append(destinationPoints, [[int(xout), int(yout)]], axis=0)

        # calculate similarity transform.
        tform = cv2.estimateAffinePartial2D(np.array([initialPoints]), np.array([destinationPoints]))
        return tform[0]


import copy
def transferExpression(lmarkSeq, meanShape):
    exptransSeq = copy.deepcopy(lmarkSeq)
    firstFlmark = exptransSeq[0,:,:]

    tformMS, _ = cv2.estimateAffine2D(firstFlmark[:,:], np.float32(meanShape[:,:]))

    sx = np.sign(tformMS[0,0])*np.sqrt(tformMS[0,0]**2 + tformMS[0,1]**2)
    sy = np.sign(tformMS[1,0])*np.sqrt(tformMS[1,0]**2 + tformMS[1,1]**2)
    # print sx, sy

    zeroVecD = np.zeros((1, 478, 2))
    diff = np.cumsum(np.insert(np.diff(exptransSeq, n=1, axis=0), 0, zeroVecD, axis=0), axis=0)
    msSeq = np.tile(np.reshape(meanShape, (1, 478, 2)), [lmarkSeq.shape[0], 1, 1])

    diff[:, :, 0] = abs(sx)*diff[:, :, 0]
    diff[:, :, 1] = abs(sy)*diff[:, :, 1]

    exptransSeq = diff + msSeq

    return exptransSeq


def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)
    
    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)])
    
    tform, _ = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
    
    return tform
    
def tformFlmarks(flmark, tform):
    transformed = np.reshape(np.array(flmark), (478, 1, 2))           
    transformed = cv2.transform(transformed, tform)
    transformed = np.float32(np.reshape(transformed, (478, 2)))
    return transformed
    
def alignEyePoints(lmarkSeq, pnt_inx=[33, 466]):
    w = 400
    h = 800

    alignedSeq = copy.deepcopy(lmarkSeq)
    firstFlmark = alignedSeq[0,:,:]
    
    eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
    eyecornerSrc  = [ (firstFlmark[pnt_inx[0], 0], firstFlmark[pnt_inx[0], 1]), 
                     (firstFlmark[pnt_inx[1], 0], firstFlmark[pnt_inx[1], 1]) ]

    tform = similarityTransform(eyecornerSrc, eyecornerDst)

    for i, lmark in enumerate(alignedSeq):
        alignedSeq[i] = tformFlmarks(lmark, tform)

    return alignedSeq

# https://anishdubey.com/face-alignment-face-landmark-images-dlib-opencv


def read_csvs(path_in, df_beh, folders=['Go1', 'Go2', "La1", "La2", "Me1", "Me2"]):
    dataset ={}
    
    for fld in folders:
        fnames = glob.glob(os.path.join(path_in, fld, '*.csv'))
        for fname in fnames:
            _, s_id = os.path.split(fname)
            subj_id = s_id[:16]
            if dataset.get(subj_id) is  None:
                dataset[subj_id] = {'ts': [], "tasks" : [], 'frame_ix':[]}
            
            ts = pd.read_csv(fname)            
            xs = ts[[ c for c in ts.columns if c.startswith('x')]]
            ys = ts[[ c for c in ts.columns if c.startswith('y')]]
            xy = np.dstack([xs,ys])
            dataset[subj_id]['frame_ix'].append(ts['frame_ix'].values)
            dataset[subj_id]['ts'].append(xy)
            dataset[subj_id]['tasks'].append(fname.split("_")[-1].split('.')[0])
            dataset[subj_id]["beh"] = df_beh.loc[subj_id]
    return dataset

    
fname_beh = '/home/adonay/Desktop/projects/Ataxia/2020_05_21_All_Tasks-data2.csv'
df_beh = pd.read_csv(fname_beh, index_col=0)
    
path_in = '/data/Dropbox (Partners HealthCare)/mediapipe_GoLaMe'
dataset = read_csvs(path_in, df_beh)


with open("inx_mp2op.pickle", "rb") as f:
    inx_op_2_mp = pickle.load(f)

pnts_inx_op = [36, 45]#, 30]

pnts_inx = [inx_op_2_mp[p] for p in pnts_inx_op]

eyes_pos = np.array([[180, 800],
            [640, 800]])

k = list(dataset)[0]



for ith, k in enumerate(list(dataset)):
    for ix, pos_raw in enumerate( dataset[k]['ts']):
        
        mask = pos_raw.sum(1).sum(1) != 0
        if pos_raw[mask].size == 0:
            print(f"subj {k} nothing in {ix}")
            dataset[k]['ts'][ix] = []
            continue
        pos_eye_fix = []
        for frame in pos_raw[mask]:        
            trans = similarityTransformMat(frame[pnts_inx[:2]], np.array(eyes_pos))
            pos_eye_fix.append(np.dot(trans, np.append(frame, np.ones([frame.shape[0],1]),1).T).T)
      
        pos_eye_fix = np.array(pos_eye_fix)
        pos_mean = np.mean(pos_eye_fix, 0)
        
        pos_deID = transferExpression(pos_eye_fix, pos_mean)
        dataset[k]['ts'][ix][mask] = pos_deID

fname = "golame_face_deID_dataset_inx_mediapipe.pickle"
with open(fname, "wb") as f:
    pickle.dump(dataset, f)


"""
    pos_alig = alignEyePoints(pos_raw, pnts_inx)
    pos_mean2 = np.mean(pos_alig, 0)
    pos_noID =  transferExpression(pos_alig, pos_mean2)
    
    
    start =  3400
    seq1 = pos_raw[start:]
    seq2 = pos_eye_fix[start:]
    seq3 = pos_noID[start:]
    
    fig, axs = plt.subplots(1,3)
    scatter1 = axs[0].scatter(seq1[0,:,0], seq1[0,:,1])
    scat1 =    axs[0].scatter(seq1[0,inx_op_2_mp,0], seq1[0,inx_op_2_mp,1], color='k') 
    scatter2 = axs[1].scatter(seq2[0,:,0], seq2[0,:,1])
    scat2 =    axs[1].scatter(seq2[0,inx_op_2_mp,0], seq2[0,inx_op_2_mp,1], color='k') 
    scatter3 = axs[2].scatter(seq3[0,:,0], seq3[0,:,1])
    scat3 =    axs[2].scatter(seq3[0,inx_op_2_mp,0], seq3[0,inx_op_2_mp,1], color='k') 
    
    
    
    [ax.invert_yaxis() for ax in axs.flat]
    
    nframes = pos_raw.shape[0]
    ttl = axs[0].set_title("{0} out of {1}")
                    
    def update(i):
        axs[0].set_title(f"{i} out of {nframes}, {(i+start)*(1/240):.2f}")
        
        scatter1.set_offsets(seq1[i])
        scat1.set_offsets(seq1[i, inx_op_2_mp])
        scatter2.set_offsets(seq2[i])
        scat2.set_offsets(seq2[i, inx_op_2_mp])
        scatter3.set_offsets(seq3[i])
        scat3.set_offsets(seq3[i, inx_op_2_mp])
        
        
    ani = animation.FuncAnimation(fig, update, frames= nframes, interval=.4)
    
    plt.show(block=False)
        
    
    fig2, axs2 = plt.subplots(1,3)
    axs2[0].plot(seq1[:,inx_op_2_mp,1])
    axs2[1].plot(seq2[:,inx_op_2_mp,1])
    axs2[2].plot(seq3[:,inx_op_2_mp,1])
    plt.show(block=False)
    
"""