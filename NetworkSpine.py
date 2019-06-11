import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from PL_with_grad_torch_HT import PermutohedralLattice as pl
   
        
config_unet = {
    "n_down": 3,
    "n_fix": 3,
    "filters_init": 8
    }


class PAM(nn.Module):
    def __init__(self, in_f, filt_f, out_f, n_split):
        super(PAM, self).__init__()
        self.in_f = in_f
        self.filt_f = filt_f
        self.out_f = out_f
        self.n_split = n_split
        self.pl = pl.apply
        self.conv_feat = nn.Conv3d(in_f + 3, filt_f*n_split, 1)
        self.conv_desc = nn.Conv3d(in_f, out_f*n_split, 1)
        self.scaling_spatial = torch.nn.Parameter(torch.randn(1, 3, 1, 1, 1))

    def forward(self, x):
        a, b, c, d, e = x.size()
        spatial_x, spatial_y, spatial_z = torch.meshgrid(torch.arange(x.size(2)).cuda(), 
                                                         torch.arange(x.size(3)).cuda(), 
                                                         torch.arange(x.size(4)).cuda())
        spatial_x = spatial_x - x.size(2)//2
        spatial_y = spatial_y - x.size(3)//2
        spatial_z = spatial_z - x.size(4)//2
        spatial = torch.stack([spatial_x, spatial_y, spatial_z], 0)
        spatial = spatial.unsqueeze(0).repeat(x.size(0), 1, 1, 1, 1)
        spatial = spatial.type(torch.cuda.FloatTensor).detach()
        spatial = spatial * self.scaling_spatial
        feat = torch.cat([x, spatial], 1)
        feat = F.leaky_relu(self.conv_feat(feat))
        feat = torch.reshape(feat, (a, self.n_split*self.filt_f, -1))
        desc = self.conv_desc(x)
        desc = torch.reshape(desc, (a, self.n_split*self.out_f, -1))
        x = torch.cat([self.pl(feat[:, k*self.filt_f:(k+1)*self.filt_f], 
                               desc[:, k*self.out_f:(k+1)*self.out_f]) for k in range(self.n_split)], 1)
        x = torch.reshape(x, (a, self.out_f*self.n_split, c, d, e))
        return x


class Unet(nn.Module):
    def __init__(self, config=config_unet, non_local=False):
        super(Unet, self).__init__()
        self.filters_init = config["filters_init"]
        self.n_fix = config["n_fix"]
        self.n_down = config["n_down"]
        self.non_local = non_local
        
        curr_f = self.filters_init
        if self.non_local:
            self.first_conv = nn.Conv3d(1, self.filters_init, 3, 1, 1)
            self.first_bn = nn.InstanceNorm3d(self.filters_init, affine=False)
        else:
            self.first_conv = nn.Conv3d(1, self.filters_init+2, 3, 1, 1)
            self.first_bn = nn.InstanceNorm3d(self.filters_init+2, affine=False)
        for j in range(self.n_fix):
            if j==0:
                if self.non_local:
                    setattr(self, "conv0_" + str(j), nn.Conv3d(curr_f, curr_f, 3, 1, 1))
                    setattr(self, "bn0_" + str(j), nn.InstanceNorm3d(curr_f, affine=False))
                else:
                    setattr(self, "conv0_" + str(j), nn.Conv3d(curr_f+2, curr_f, 3, 1, 1))
                    setattr(self, "bn0_" + str(j), nn.InstanceNorm3d(curr_f, affine=False))
            else:
                setattr(self, "conv0_" + str(j), nn.Conv3d(curr_f, curr_f, 3, 1, 1))
                setattr(self, "bn0_" + str(j), nn.InstanceNorm3d(curr_f, affine=False))
        for i in range(self.n_down):
            curr_f *= 2
            setattr(self, "conv_down" + str(i), nn.Conv3d(curr_f//2, curr_f, 3, 2, 1))
            setattr(self, "bn_down_" + str(i), nn.InstanceNorm3d(curr_f, affine=False))
            for j in range(self.n_fix):
                setattr(self, 
                        "conv" + str(i+1) + "_" + str(j), 
                        nn.Conv3d(curr_f, curr_f, 3, 1, 1))
                setattr(self, 
                        "bn" + str(i+1) + "_" + str(j), 
                        nn.InstanceNorm3d(curr_f, affine=False))
        for i in range(self.n_down):
            curr_f = curr_f//2
            setattr(self, 
                    "conv_up" + str(i), 
                    nn.ConvTranspose3d(curr_f*2, curr_f, 4, 2, 1))
            setattr(self, "bn_up_" + str(i), nn.InstanceNorm3d(curr_f, affine=False))
            for j in range(self.n_fix):
                if j==0:
                    setattr(self, 
                            "conv" + str(i+self.n_down+1) + "_" + str(j), 
                            nn.Conv3d(2*curr_f, curr_f, 3, 1, 1))
                    setattr(self, 
                            "bn" + str(i+self.n_down+1) + "_" + str(j), 
                            nn.InstanceNorm3d(curr_f, affine=False))
                else:
                    setattr(self, 
                            "conv" + str(i+self.n_down+1) + "_" + str(j), 
                            nn.Conv3d(curr_f, curr_f, 3, 1, 1))
                    setattr(self, 
                            "bn" + str(i+self.n_down+1) + "_" + str(j), 
                            nn.InstanceNorm3d(curr_f, affine=False))
        self.conv_f = nn.Conv3d(curr_f, 25, 1)
        if non_local:
            self.pam = PAM(self.filters_init*2, self.filters_init, self.filters_init, 2)
        
    def forward(self, x):
        x = F.relu(self.first_bn(self.first_conv(x)))
        res_int = []
        for j in range(self.n_fix):
            x = getattr(self, "conv0_" + str(j))(x)
            x = getattr(self, "bn0_" + str(j))(x)
            x = F.relu(x)
        res_int.append(x)
        for i in range(self.n_down):
            x = getattr(self, "conv_down" + str(i))(x)
            x = getattr(self, "bn_down_" + str(i))(x)
            x = F.relu(x)
            for j in range(self.n_fix):
                x = getattr(self, "conv" + str(1+i) + "_" + str(j))(x)
                x = getattr(self, "bn" + str(1+i) + "_" + str(j))(x)
                x = F.relu(x)
            res_int.append(x)
        for i in range(self.n_down):
            x = getattr(self, "conv_up" + str(i))(x)
            x = getattr(self, "bn_up_" + str(i))(x)
            x = F.relu(x)
            x = torch.cat([x, res_int[-2-i]], dim=1)
            for j in range(self.n_fix):
                x = getattr(self, "conv" + str(i+self.n_down+1) + "_" + str(j))(x)
                x = getattr(self, "bn" + str(i+self.n_down+1) + "_" + str(j))(x)
                x = F.relu(x)
            if self.non_local:
                if i==(self.n_down-2):
                    x = self.pam(x)
        x = self.conv_f(x)
        x = F.softmax(x, dim=1)
        return x