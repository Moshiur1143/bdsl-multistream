"""V2 experiment: joint-aware graph + ordered temporal representation."""
import torch
from torch import nn
from .model import GraphBlock

class GraphV2(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.embed=nn.Parameter(torch.randn(1,16,1,75)*.02)
        self.blocks=nn.ModuleList([GraphBlock(26,64),GraphBlock(64,96)])
        # Explicit joint-order projection retains which finger/hand moved.
        self.ordered=nn.Sequential(nn.Conv1d(75*10,192,1),nn.GroupNorm(8,192),nn.GELU())
        self.project=nn.Conv1d(96*3+192,256,1)
        self.temporal=nn.ModuleList([nn.Sequential(nn.Conv1d(256,256,5,padding=2*d,dilation=d),
            nn.GroupNorm(8,256),nn.GELU(),nn.Dropout(.2)) for d in (1,2,4)])
        self.attention=nn.Conv1d(256,1,1)
        self.head=nn.Sequential(nn.LayerNorm(512),nn.Dropout(.35),nn.Linear(512,num_classes))
    def forward(self,graph,mask,frames):
        mask=mask.bool()&frames[:,:,None].bool();m=mask[:,None].float()
        graph=graph*mask[...,None]; b,t=graph.shape[:2]
        ordered=self.ordered(graph.reshape(b,t,-1).transpose(1,2))
        x=torch.cat([graph.permute(0,3,1,2),self.embed.expand(b,-1,t,-1)],1)*m
        for block in self.blocks:x=block(x,m)
        groups=[]
        for sl in (slice(0,21),slice(21,42),slice(42,67)):
            groups.append((x[...,sl]*m[...,sl]).sum(-1)/m[...,sl].sum(-1).clamp_min(1))
        valid=mask.any(-1);v=valid[:,None]
        x=self.project(torch.cat([*groups,ordered],1))*v
        for block in self.temporal:x=(x+block(x))*v
        scores=self.attention(x).masked_fill(~v,-1e4)
        attention=torch.softmax(scores,-1)*v;attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled=(x*attention).sum(-1)
        maximum=x.masked_fill(~v,-1e4).amax(-1);maximum=torch.where(valid.any(-1)[:,None],maximum,torch.zeros_like(maximum))
        return self.head(torch.cat([pooled,maximum],-1))

class MultiStreamSignModelV2(nn.Module):
    def __init__(self,num_classes=200,mode='graph_v2',pretrained_face=True):
        super().__init__();assert num_classes>=150
        self.mode=mode;self.graph=GraphV2(num_classes)
        if mode=='fusion_v2':
            from torchvision.models import resnet18,ResNet18_Weights
            from pathlib import Path
            torch.hub.set_dir(str(Path(__file__).resolve().parents[1]/'models/pretrained'))
            backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_face else None)
            self.face_encoder=nn.Sequential(*list(backbone.children())[:-1])
            self.face_encoder.requires_grad_(False)
            self.face_temporal=nn.Sequential(nn.Conv1d(512,128,3,padding=1),nn.GELU(),nn.Dropout(.3))
            self.face_head=nn.Linear(128,num_classes)
            self.strength=nn.Parameter(torch.tensor(-3.))
            self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
            self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        elif mode!='graph_v2':raise ValueError(mode)
    def train(self,mode=True):
        super().train(mode)
        if self.mode=='fusion_v2':self.face_encoder.eval();self.graph.eval()
        return self
    def forward(self,graph,graph_mask,face,face_mask,frame_mask):
        logits=self.graph(graph,graph_mask,frame_mask)
        if self.mode=='graph_v2':return logits
        ids=torch.linspace(0,face.shape[1]-1,8,device=face.device).round().long()
        mask=(face_mask&frame_mask)[:,ids];b=face.shape[0]
        with torch.no_grad():
            img=(face[:,ids].reshape(b*8,3,96,96)+1)/2
            img=torch.nn.functional.interpolate(img,size=(224,224),mode='bilinear',align_corners=False)
            img=(img-self.mean)/self.std
            f=self.face_encoder(img).flatten(1).reshape(b,8,512).transpose(1,2)*mask[:,None]
        f=self.face_temporal(f)*mask[:,None]
        f=f.sum(-1)/mask.sum(-1,keepdim=True).clamp_min(1)
        delta=self.face_head(f)*mask.any(-1)[:,None]
        return logits+torch.sigmoid(self.strength)*delta
