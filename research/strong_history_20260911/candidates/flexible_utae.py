"""U-TAE adapted to masked LST regression, independent of the project network.

The upstream convolutional encoder, L-TAE, and decoder are reused under MIT.
Task adaptations: separate current/history input stems, spatial visibility in
temporal attention and skip aggregation, signed regression, support projection.
"""
import torch
from torch import nn
from torch.nn import functional as F
from historylst.third_party.utae.utae import UTAE

def project(p, coarse, support):
    p=p.float();m=support.float()
    fraction=F.avg_pool2d(m,4)
    mean=F.avg_pool2d(p*m,4)/fraction.clamp_min(1/16)
    delta=torch.where(torch.isfinite(coarse)&(fraction>0),coarse.float()-mean,0.)
    return (p+F.interpolate(delta,scale_factor=4,mode='nearest'))*m

def stem(channels):
    return nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU())

class FlexibleUTAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.current=stem(74);self.historical=stem(9)
        self.core=UTAE(input_dim=32,encoder_widths=[32,64,128,256],decoder_widths=[32,64,128,256],
            out_conv=[32,1],n_head=8,d_model=256,d_k=4,pad_value=None)
        self.core.out_conv=nn.Conv2d(32,1,1)
        nn.init.zeros_(self.core.out_conv.weight);nn.init.zeros_(self.core.out_conv.bias)

    def forward(self,fine,coarse,support,context,emissivity,history):
        b,_,h,w=fine.shape
        n=history.shape[1]
        base=fine[:,:1].float()
        fine=torch.cat(((base-300)/20,fine[:,1:].float()),1)
        cvalid=torch.isfinite(coarse)
        cm=F.interpolate(torch.cat((torch.nan_to_num((coarse.float()-300)/20),cvalid.float()),1),size=(h,w),mode='nearest')
        hist=history.float().clone();em=emissivity.float()
        if self.training:
            keep=(torch.rand(b,1,1,1,1,device=hist.device)>=.25)
            hist=hist*keep
            em=em*(torch.rand(b,1,1,1,device=em.device)>=.25)
        thermal=(hist[:,:,2:3]>0);emis=(hist[:,:,5:6]>0)
        hist[:,:,:2]=hist[:,:,:2]*thermal
        hist[:,:,3:4]=hist[:,:,3:4]*thermal
        hist[:,:,4:5]=hist[:,:,4:5]*emis
        curr=self.current(torch.cat((fine,em,support.float(),cm,context.float()[:,:,None,None].expand(-1,-1,h,w)),1))
        past=self.historical(hist.reshape(b*n,9,h,w)).reshape(b,n,32,h,w) if n else hist.new_empty(b,0,32,h,w)
        x=torch.cat((curr[:,None],past),1)
        visibility=torch.cat((torch.ones(b,1,h,w,device=x.device),thermal.squeeze(2).float()),1)
        # Current query token contains predictors only, never query fine LST.
        positions=torch.cat((hist.new_zeros(b,1),-hist[:,:,8].flatten(2).amax(2)*3652.5),1)
        maps=[self.core.in_conv.smart_forward(x)]
        for block in self.core.down_blocks:maps.append(block.smart_forward(maps[-1]))
        lowmask=F.adaptive_max_pool2d(visibility,maps[-1].shape[-2:])==0
        out,att=self.core.temporal_encoder(maps[-1],batch_positions=positions,pad_mask=lowmask)
        heads=att.shape[0]
        for i,up in enumerate(self.core.up_blocks):
            features=maps[-i-2];shape=features.shape[-2:]
            a=F.interpolate(att.reshape(heads*b,n+1,*att.shape[-2:]),size=shape,mode='bilinear',align_corners=False).reshape(heads,b,n+1,*shape)
            visible=F.adaptive_max_pool2d(visibility,shape)
            a=a*visible[None];a=a/a.sum(2,keepdim=True).clamp_min(1e-6)
            grouped=torch.stack(features.chunk(heads,dim=2))
            skip=(a[:,:,:,None]*grouped).sum(2)
            skip=torch.cat(list(skip),dim=1)
            out=up(out,skip)
        residual=self.core.out_conv(out).float()
        return project(base+residual,coarse,support)
