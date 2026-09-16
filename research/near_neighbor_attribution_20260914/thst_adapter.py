"""Pinned THSTNet two-stage core with explicit common-input embeddings."""
from pathlib import Path
import sys
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'vendor_deps'),str(HERE/'vendor/thstnet'),str(HERE.parents[1]/'resources/historylst246')]
from stage_one import stage_one
from stage_two import stage_two
from historylst.model import project


class FP32Mapping(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
    def forward(self,c0,c1,f0):
        with torch.autocast(c0.device.type,enabled=False):
            return self.original(c0.float(),c1.float(),f0.float())


class AuxiliaryEmbedding(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        self.aux_projection=nn.Conv2d(155,32,2,stride=2)
        nn.init.zeros_(self.aux_projection.weight);nn.init.zeros_(self.aux_projection.bias)
        self.aux=None
    def forward(self,x):
        if self.aux is None:raise RuntimeError('Explicit current/history auxiliary input required')
        return self.original(x)+self.aux_projection(self.aux).flatten(2).transpose(1,2)


def auxiliary(batch):
    fine=batch['fine'].float();fine=torch.cat(((fine[:,:1]-300)/20,fine[:,1:]),1)
    coarse=batch['coarse'].float()
    cm=torch.nn.functional.interpolate(torch.cat((torch.nan_to_num((coarse-300)/20),torch.isfinite(coarse).float()),1),
                                        size=fine.shape[-2:],mode='nearest')
    hist=batch['history'].float().clone()
    visible=hist[:,:,2:3]>0
    hist[:,:,:2]=torch.where(visible,hist[:,:,:2],0.)
    hist[:,:,3:4]=torch.where(visible,hist[:,:,3:4],0.)
    hist[:,:,4:5]=torch.where(hist[:,:,5:6]>0,hist[:,:,4:5],0.)
    ctx=batch['context'].float()[:,:,None,None].expand(-1,-1,*fine.shape[-2:])
    out=torch.cat((fine,batch['emissivity'].float(),batch['support'].float(),cm,ctx,
                   hist.flatten(1,2)),1)
    if out.shape[1]!=155:raise ValueError('Six-field information adapter changed')
    return out


class CommonInputTHST(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=stage_one(patchsize=128,in_dim=32)
        self.second=stage_two(patchsize=128,in_dim=32)
        self.first.stage_one.ST_mapping=FP32Mapping(self.first.stage_one.ST_mapping)
        self.first.stage_one.PatchEmbed=AuxiliaryEmbedding(self.first.stage_one.PatchEmbed)
        self.second.stage2.c_down.PatchEmbed=AuxiliaryEmbedding(self.second.stage2.c_down.PatchEmbed)
        self.second.stage2.f_down.PatchEmbed=AuxiliaryEmbedding(self.second.stage2.f_down.PatchEmbed)
        self.stage=1

    def set_stage(self,stage):
        if stage not in (1,2):raise ValueError(stage)
        self.stage=stage
        self.first.requires_grad_(stage==1)
        self.second.requires_grad_(stage==2)

    def forward(self,c0,f0,c1,aux):
        first_embedding=self.first.stage_one.PatchEmbed
        second_embeddings=[self.second.stage2.c_down.PatchEmbed,self.second.stage2.f_down.PatchEmbed]
        first_embedding.aux=aux
        for layer in second_embeddings:layer.aux=aux
        try:
            if self.stage==1:
                _,_,output=self.first(c0,f0,c1)
            else:
                self.first.eval()
                with torch.no_grad():down,up,_=self.first(c0,f0,c1)
                output=self.second(c0,f0,c1,down,up)
            return output
        finally:
            first_embedding.aux=None
            for layer in second_embeddings:layer.aux=None


def prediction_kelvin(model,batch,c0,f0,c1):
    normalized=model(c0,f0,c1,auxiliary(batch))
    return project(100*normalized.float()+250,batch['coarse'],batch['support'])
