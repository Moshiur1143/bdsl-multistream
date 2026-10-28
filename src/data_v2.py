"""V2: preserve joint identity and local 3D hand shape; reuse cached detections."""
import numpy as np,torch
from torch.utils.data import Dataset
from .data import normalize

FEATURE_VERSION='joint_order_local_xyz_v2'
def normalize_v2(a):
    result=normalize(a); mask=result['graph_mask']; graph=result['graph']
    local=np.zeros((*mask.shape,3),np.float32)
    local[:,:42]=np.clip(a['hand_local_xyz'].reshape(len(mask),42,3),-5,5)
    local*=mask[...,None]
    # Parent-relative finger direction in body-normalized image coordinates.
    parents=[0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]
    bone=np.zeros((*mask.shape,2),np.float32)
    for offset in (0,21):
        for j,parent in enumerate(parents):
            valid=mask[:,offset+j]&mask[:,offset+parent]
            bone[:,offset+j]=(graph[:,offset+j,:2]-graph[:,offset+parent,:2])*valid[:,None]
    result['graph']=np.concatenate([graph,local,bone],-1)
    return result

class FeatureDatasetV2(Dataset):
    def __init__(self,rows,root,augment=False):self.rows=rows.reset_index(drop=True);self.root=root;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]
        with np.load(self.root/row.feature_path,allow_pickle=False) as z:
            a={k:z[k] for k in ('graph_xy','graph_mask','face_rgb','face_mask','frame_mask','unique_frame_mask','hand_local_xyz')}
        if self.augment and np.random.random()<.5:
            # Small temporal crop, identical indices for both streams; never reverse signs.
            left=np.random.randint(0,3);right=np.random.randint(29,33)
            ids=np.rint(np.linspace(left,right-1,32)).astype(int)
            a={k:v[ids] for k,v in a.items()}
            a['unique_frame_mask']=a['unique_frame_mask'] & np.r_[True,np.diff(ids)>0]
        x=normalize_v2(a)
        if self.augment:
            scale=np.random.uniform(.95,1.05);x['graph'][...,:4]*=scale;x['graph'][...,8:]*=scale
        return {k:torch.from_numpy(v.copy()) for k,v in x.items()},int(row.class_id)
