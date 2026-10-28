"""Shared feature normalization for training and live inference."""
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize(arrays):
    xy=arrays['graph_xy'].astype(np.float32).copy()
    mask=arrays['graph_mask'].astype(bool).copy()
    frames=arrays['frame_mask'].astype(bool)&arrays['unique_frame_mask'].astype(bool)
    mask &= frames[:,None]
    # Use visible shoulders, otherwise visible upper-body points; no dataset fitting.
    for t in range(len(xy)):
        if mask[t,53] and mask[t,54]:
            center=(xy[t,53]+xy[t,54])/2
            scale=max(float(np.linalg.norm(xy[t,53]-xy[t,54])),.05)
        else:
            visible=xy[t,42:67][mask[t,42:67]]
            center=visible.mean(0) if len(visible) else np.array([.5,.5])
            scale=.25
        xy[t]=(xy[t]-center)/scale
    xy=np.clip(xy,-10,10)*mask[...,None]
    velocity=np.zeros_like(xy)
    # Per-sampled-step motion (not claimed to be physical velocity).
    velocity[1:]=(xy[1:]-xy[:-1])*(mask[1:]&mask[:-1])[...,None]
    graph=np.concatenate([xy,velocity,mask[...,None].astype(np.float32)],-1)
    face=arrays['face_rgb'].astype(np.float32)/255
    face_mask=arrays['face_mask'].astype(bool)&frames
    face=((face-.5)/.5)*face_mask[:,None,None,None]
    return dict(graph=graph,graph_mask=mask,face=face.transpose(0,3,1,2),
                face_mask=face_mask,frame_mask=frames)

class FeatureDataset(Dataset):
    def __init__(self,rows,root,augment=False):
        self.rows=rows.reset_index(drop=True); self.root=root; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]
        with np.load(self.root/row.feature_path,allow_pickle=False) as z:
            a={k:z[k] for k in ('graph_xy','graph_mask','face_rgb','face_mask','frame_mask','unique_frame_mask')}
        x=normalize(a)
        if self.augment:
            # Small shared coordinate scale perturbation; no mirroring or sign reversal.
            x['graph'][...,:4]*=np.random.uniform(.95,1.05)
            brightness=np.random.uniform(.9,1.1)
            x['face']=np.clip(x['face']*brightness,-1,1)
        return {k:torch.from_numpy(v.copy()) for k,v in x.items()},int(row.class_id)
