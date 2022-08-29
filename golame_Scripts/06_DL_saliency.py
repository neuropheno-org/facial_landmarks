import torch
import numpy as np
import time
from datetime import datetime
import matplotlib.cm as cm
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchaudio.transforms as T
from DL_utils import (get_data, MyDataset, MyDataset_2data, inception_model, FCNN_short, binary_acc,
                      CNN_short_fc, multiinception_model, CNN_short_face_fc, inception_model1, CNN_short_fc_multi)

model_fname = 'checkpoints/inception_face_100wz_50st_AtaxiavsControl_mask_y_len10_mask_x_len10_lr1e-05_derivTrue_pntsubsetTrue_optimAdam_valBatch8_22-07-12_21-53-05'
model_fname = 'checkpoints/inception_face_transferlearningfixW0_100wz_50st_AtaxiavsControl_mask_y_len10_mask_x_len10_lr0.0001_derivTrue_pntsubsetTrue_optimAdam_valBatch8_22-07-27_14-09-55'
net = torch.load(model_fname)


win_len = 100
hop = 50
vad_min = .5
g1, g0 = 'Ataxia', 'Control'
batch_train = 8
batch_val = 8  # all if -1
dev_num = 2
data_input =  "face" #  "multi" #"audio"  #  
n_epochs = 300
split = 'individ'
optim = "Adam"  # "SGD"#
mask_y_len = 10
mask_x_len = 10
lr = 1e-4
model_name =  "inception"  # "CNN_short_face_fc" # "CNN_short_fc" # "CNN_short_fc_multi" #  "inception_model1" #   "inception_multi"  # "FCNN_short" #  
pose_pnts_small =   True #"eyes_mouse"# "mouse" #  
derivatives= True
save = True


if  not 'inception' in model_name:
    audio_dims = 3
    pose_dims = 3
else:
    audio_dims = 2
    pose_dims = 2
    

params_string = f"model_name = {model_name} \ndata_input =  {data_input} \nwin_len = {win_len} \nhop = {hop} \noptim = {optim} \nlr={lr} \nmask_y_len = {mask_y_len} mask_x_len = {mask_x_len} \
\nvad_min = {vad_min}  \ng1, g0 = {g1} {g0} \nbatch_train = {batch_train} batch_val = {batch_val}  \nn_epochs = {n_epochs} \
\nsplit = {split} \nderivatives = {derivatives} \npose_pnts_small = {pose_pnts_small}"
print(params_string)

# Get dataset
fname = "golame_face_deID_dataset_inx_mediapipe.pickle"

with open(fname, "rb") as f:
    dataset = pickle.load(f)

if pose_pnts_small:
    # reduce num points face
    with open("inx_mp2op.pickle", "rb") as f:
        pnts_inx = pickle.load(f)
    if pose_pnts_small == "eyes_mouse":
        subset = np.arange(36, 70)
        pnts_inx = np.array(pnts_inx)[subset]
    elif pose_pnts_small == "mouse":
        subset = np.arange(48, 68)
        pnts_inx = np.array(pnts_inx)[subset]
    for k in list(dataset):
        dataset[k]['ts'] = [d[:, pnts_inx, :] if len(d) else d for d in dataset[k]['ts']]

# Remove other groups
ids = list(dataset)
for k in ids:
    if dataset[k]['beh']['gen_diagnosis'] not in [g1, g0]:
        del dataset[k]

# Split data into groups
ids = np.array(list(dataset))
labels = [dataset[k]['beh']['gen_diagnosis'] for k in ids]
labels = [1 if l == g1 else 0 for l in labels]


if split == "strat":
    train_inx, val_inx = train_test_split(list(range(len(labels))), test_size=.3,
                                        random_state=1, stratify=labels)
elif split == "individ":
    IDs = [i[:5] for i in list(dataset.keys())]
    idxs, inx, groups = np.unique(IDs, return_index=True, return_inverse=True)
    gss = GroupShuffleSplit(1, train_size=.8, random_state=1)
    train_inx, val_inx = next(
        gss.split(list(range(len(labels))), None, groups))

data_train = {k: v for k, v in dataset.items() if k in ids[train_inx]}
data_val = {k: v for k, v in dataset.items() if k in ids[val_inx]}

labels = np.array(labels)

l1 = f"subj data train: {len(data_train)}, {g1}:{sum(labels[train_inx]== 1)},{g0}:{sum(labels[train_inx]== 0)}"
l2 = f"subj data val: {len(data_val)}, {g1}:{sum(labels[val_inx]== 1)},{g0}:{sum(labels[val_inx]== 0)}"
print(l1)
print(l2)

x_ts_train, x_S_train, y_train, sbj_ids_train, inx_win, task_ids_t = get_data(
    data_train, win_len, hop, vad_min=None, pose_dims=pose_dims, audio_dims=None, return_tasks=True)
x_ts_val, x_S_val, y_val, sbj_ids_val, n, task_ids_v = get_data(
    data_val, win_len, hop, vad_min=None, pose_dims=pose_dims,  audio_dims=None, return_tasks=True)

y_train = [1 if l == g1 else 0 for l in y_train]
y_val = [1 if l == g1 else 0 for l in y_val]

val_1 = sum(y_val)
val_0 = abs(val_1 - len(y_val))

l3 = f"data train: {len(x_ts_train)}, {g1}:{sum(y_train)},{g0}:{abs(sum(y_train) - len(y_train))} "
l4 = f"data val: {len(x_ts_val)}, {g1}:{val_1},{g0}:{val_0}, ratio:{val_0/val_1 :.2f} "
print(l3)
print(l4)

# Create Dataloaders
transform = transforms.Compose([
    T.TimeMasking(time_mask_param=mask_x_len),
    T.FrequencyMasking(freq_mask_param=mask_y_len)
])

if data_input == "face":
    dataset_t = MyDataset(x_ts_train, y_train, transform=transform)
    dataset_v = MyDataset(x_ts_val, y_val, transform=transform)
elif data_input == "audio":
    dataset_t = MyDataset(x_S_train, y_train, transform=transform)
    dataset_v = MyDataset(x_S_val, y_val, transform=transform)
elif "multi" in data_input:
    dataset_t = MyDataset_2data(x_ts_train, x_S_train, y_train, transform=transform)
    dataset_v = MyDataset_2data(x_ts_val, x_S_val, y_val, transform=transform)

dataloader_t = DataLoader(dataset_t, batch_size=batch_train,
                        shuffle=True, drop_last=True, pin_memory=True)
batch_size = 1
dataloader_v = DataLoader(
    dataset_v, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "cuda" and dev_num is not None:
    device = torch.device(f'cuda:{dev_num}')


#we don't need gradients w.r.t. weights for a trained model
for param in net.parameters():
    param.requires_grad = False

#set model in eval mode
net.eval()

preds = []
lbs = []
saliences = []
for i, data in enumerate(dataloader_v):
    if "multi" in data_input:
        inputs1, inputs2, labels = data
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
    else:
        inputs, labels = data
        inputs = inputs.to(device)
    labels = labels.to(device)

    labels = labels.type(torch.long)
    labels = labels.unsqueeze(1)

    inputs.requires_grad_()
    # forward
    if "multi" in data_input:
        outputs = net(inputs1, inputs2)
    else:
        outputs = net(inputs)

    pred = F.softmax(outputs, dim=1).argmax(1).cpu()
    preds.extend(pred)    
    lbs.extend(labels[:, 0].cpu())

    score, indices = torch.max(outputs, 1)

    # backward pass to get gradients of score predicted class w.r.t. input image
    score[0].backward()
    slc = torch.abs(inputs.grad[0])
    # slc = (slc - slc.min())/(slc.max()-slc.min())
    slc = slc.cpu()
    
    saliences.append(slc)
    
saliences = np.stack(saliences)
preds = np.stack(preds)
lbs = np.stack(lbs)

correct = preds == lbs

labels_v = [dataset[k]['beh']['gen_diagnosis'] for k in ids[val_inx]]
task_id = ["La" in t for t in task_ids_v]
msk = correct #np.arange(len(saliences))#   task_id # 
# fig, axs = plt.subplots(3,2)
# axs[0, 0].plot(saliences[msk].mean(0)[:70])
# axs[0, 1].plot(saliences[msk].mean(0)[70:])
# axs[1, 0].plot(saliences[msk].std(0)[:70])
# axs[1, 1].plot(saliences[msk].std(0)[70:])
# axs[2, 0].imshow(saliences[msk].mean(0))
# # axs[2, 1].imshow(saliences[msk].mean(0)[1])
# plt.show(block=False)

ss = saliences[msk].mean(0).mean(1)
pnts_pos = dataset[list(dataset)[3]]['ts'][0].mean(0)

fig, axs = plt.subplots(1,3, figsize=(20,10))
st = axs[0].scatter(pnts_pos[:,0], pnts_pos[:,1], c=ss[:70], cmap=cm.jet)
axs[0].invert_yaxis()
fig.colorbar(st, ax=axs[0])
st = axs[1].scatter(pnts_pos[:,0], pnts_pos[:,1], c=ss[70:], cmap=cm.jet)
axs[1].invert_yaxis()
fig.colorbar(st, ax=axs[1])
sn = np.sqrt(ss[:70]**2 + ss[70:]**2)
st = axs[2].scatter(pnts_pos[:,0], pnts_pos[:,1], c=sn, cmap=cm.jet)
axs[2].invert_yaxis()
fig.colorbar(st, ax=axs[2])
plt.show(block=False)