"""Small temporal graph + face model. All outputs are class logits."""
import torch
from torch import nn

HAND_EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
              (13,17),(0,17),(17,18),(18,19),(19,20)]
POSE_EDGES = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
              (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
              (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
              (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
              (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]

def adjacency():
    a=torch.eye(75)
    edges=HAND_EDGES+[(i+21,j+21) for i,j in HAND_EDGES]
    edges += [(i+42,j+42) for i,j in POSE_EDGES]+[(0,57),(21,58)]
    for i,j in edges: a[i,j]=a[j,i]=1
    return a/a.sum(-1,keepdim=True)

class GraphBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.register_buffer('a',adjacency())
        self.spatial=nn.Conv2d(cin,cout,1)
        self.temporal=nn.Sequential(nn.GroupNorm(8,cout),nn.SiLU(),
            nn.Conv2d(cout,cout,(5,1),padding=(2,0)),nn.GroupNorm(8,cout),nn.Dropout(.15))
        self.res=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,mask):
        # Mask before propagation; invalid stored values cannot enter the graph.
        x=x*mask
        mixed=torch.einsum('bctv,vw->bctw',x,self.a)
        return torch.nn.functional.silu(self.temporal(self.spatial(mixed))+self.res(x))*mask

class GraphStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([GraphBlock(5,64),GraphBlock(64,96),GraphBlock(96,128)])
    def forward(self,x,mask):
        x=x.permute(0,3,1,2); m=mask[:,None].to(x.dtype)
        for block in self.blocks: x=block(x,m)
        return (x*m).sum(-1)/m.sum(-1).clamp_min(1)

class FaceStream(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]; cin=3
        for cout in (24,48,96,128):
            layers += [nn.Conv2d(cin,cout,3,stride=2,padding=1),nn.GroupNorm(8,cout),nn.SiLU()]
            cin=cout
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.temporal=nn.Sequential(nn.Conv1d(128,128,5,padding=2),nn.GELU(),nn.Dropout(.2))
    def forward(self,face,mask):
        b,t=face.shape[:2]
        face=face*mask[:,:,None,None,None]
        x=self.encoder(face.reshape(b*t,3,96,96)).reshape(b,t,128).transpose(1,2)
        x=x*mask[:,None]
        return self.temporal(x)*mask[:,None]

class MultiStreamSignModel(nn.Module):
    def __init__(self,num_classes=200,mode='fusion'):
        super().__init__()
        if num_classes<150: raise ValueError('This project requires at least 150 classes')
        if mode not in ('graph','face','fusion'): raise ValueError(mode)
        self.num_classes=num_classes; self.mode=mode
        self.graph=GraphStream() if mode!='face' else None
        self.face=FaceStream() if mode!='graph' else None
        self.gate=nn.Sequential(nn.Conv1d(258,64,1),nn.GELU(),nn.Conv1d(64,2,1)) if mode=='fusion' else None
        self.head=nn.Sequential(nn.LayerNorm(128),nn.Dropout(.3),nn.Linear(128,num_classes))
    def forward(self,graph,graph_mask,face,face_mask,frame_mask):
        gm=graph_mask.bool() & frame_mask[:,:,None].bool()
        fm=face_mask.bool() & frame_mask.bool()
        g=self.graph(graph,gm) if self.graph is not None else None
        f=self.face(face,fm) if self.face is not None else None
        gv=gm.any(-1)
        if self.mode=='graph': x=g; valid=gv
        elif self.mode=='face': x=f; valid=fm
        else:
            qualities=torch.stack([gm.float().mean(-1),fm.float()],1)
            scores=self.gate(torch.cat([g,f,qualities],1))
            available=torch.stack([gv,fm],1)
            weights=torch.softmax(scores.masked_fill(~available,-1e4),dim=1)*available
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-6)
            x=g*weights[:,0:1]+f*weights[:,1:2]; valid=gv|fm
        pooled=(x*valid[:,None]).sum(-1)/valid.sum(-1,keepdim=True).clamp_min(1)
        return self.head(pooled)
