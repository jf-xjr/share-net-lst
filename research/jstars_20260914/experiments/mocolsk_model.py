"""Published four-stage MoCoLSK, with the common six-field input contract."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'resources/historylst246'),
                str(ROOT/'research/near_neighbor_attribution_20260914')]
from train_mocolsk_v2 import import_model, patch_dynamic_mlp_gradients
from historylst.model import project


def auxiliary(batch):
    fine = batch['fine'].float()
    fine = torch.cat(((fine[:,:1]-300)/20, fine[:,1:]), 1)
    coarse = batch['coarse'].float()
    cm = F.interpolate(torch.cat((torch.nan_to_num((coarse-300)/20),
                                   torch.isfinite(coarse).float()), 1),
                       size=fine.shape[-2:], mode='nearest')
    hist = batch['history'].float().clone()
    visible = hist[:,:,2:3] > 0
    hist[:,:,:2] = torch.where(visible, hist[:,:,:2], 0.)
    hist[:,:,3:4] = torch.where(visible, hist[:,:,3:4], 0.)
    hist[:,:,4:5] = torch.where(hist[:,:,5:6] > 0, hist[:,:,4:5], 0.)
    ctx = batch['context'].float()[:,:,None,None].expand(-1,-1,*fine.shape[-2:])
    out = torch.cat((fine, batch['emissivity'].float(), batch['support'].float(),
                     cm, ctx, hist.flatten(1,2)), 1)
    assert out.shape[1] == 155
    return out


class CommonMoCoLSK(nn.Module):
    def __init__(self, repair_gradient=True):
        super().__init__()
        self.backbone = import_model()(in_channels=1, gui_channels=155, scale=4,
                                       num_feats=32, n_resblocks=4, num_stages=4)
        if repair_gradient:
            patch_dynamic_mlp_gradients(self.backbone)

    def forward(self, fine, coarse, support, context, emissivity, history):
        fields = dict(fine=fine, coarse=coarse, support=support, context=context,
                      emissivity=emissivity, history=history)
        local_fill = F.avg_pool2d(fine[:,:1].float(), 4, 4)
        normalized = (torch.where(torch.isfinite(coarse), coarse.float(), local_fill)-300)/20
        output = self.backbone(normalized, auxiliary(fields))
        return project(output.float()*20+300, coarse, support)
