import torch.nn as nn
import torch.nn.functional as F
import torch

from pl import PermutohedralLattice as pl

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