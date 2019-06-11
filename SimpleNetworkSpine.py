import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from PL_with_grad_torch_HT import PermutohedralLattice as pl


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

    
class SimpleNet(nn.Module):
    def __init__(self, non_local=False, dilated=False):
        super(SimpleNet, self).__init__()
        self.non_local = non_local
        self.dilated = dilated
        n_f = 18
        if non_local:
            self.conv_init = nn.Sequential(
                                            nn.Conv3d(1, n_f, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f),
                                            nn.ReLU()
                                            )
        else:
            self.conv_init = nn.Sequential(
                                            nn.Conv3d(1, n_f + 2, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f + 2),
                                            nn.ReLU()
                                            )
        if dilated:
            if non_local:
                self.conv1_0 = nn.Sequential(
                                            nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
                self.conv1_1 = nn.Sequential(
                                            nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
                self.conv1_2 = nn.Sequential(
                                            nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
            else:
                self.conv1_0 = nn.Sequential(
                                            nn.Conv3d(n_f + 2, n_f//3, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
                self.conv1_1 = nn.Sequential(
                                            nn.Conv3d(n_f + 2, n_f//3, 3, 1, 2, dilation=2),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
                self.conv1_2 = nn.Sequential(
                                            nn.Conv3d(n_f + 2, n_f//3, 3, 1, 4, dilation=4),
                                            nn.InstanceNorm3d(n_f//3),
                                            nn.ReLU()
                                            )
            self.conv2_0 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv2_1 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv2_2 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv3_0 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv3_1 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv3_2 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv4_0 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv4_1 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
            self.conv4_2 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                           nn.InstanceNorm3d(n_f//3),
                                           nn.ReLU()
                                          )
        else:
            if non_local:
                self.conv1 = nn.Sequential(
                                            nn.Conv3d(n_f, n_f, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f),
                                            nn.ReLU()
                                            )
            else:
                    self.conv1 = nn.Sequential(
                                            nn.Conv3d(n_f + 2, n_f, 3, 1, 1),
                                            nn.InstanceNorm3d(n_f),
                                            nn.ReLU()
                                            )
            self.conv2 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f),
                                           nn.ReLU()
                                          )
            self.conv3 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f),
                                           nn.ReLU()
                                          )
            if self.non_local:
                self.pam = PAM(n_f, n_f//2, n_f//2, 2)
            self.conv4 = nn.Sequential(
                                           nn.Conv3d(n_f, n_f, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f),
                                           nn.ReLU()
                                          )
        self.segm = nn.Conv3d(n_f, 25, 1)
    
        
    def forward(self, x):
        x = self.conv_init(x) 
        if self.dilated:
            x_0 = self.conv1_0(x) 
            x_1 = self.conv1_1(x) 
            x_2 = self.conv1_2(x)
            x = torch.cat([x_0, x_1, x_2], 1)
            x_0 = self.conv2_0(x) 
            x_1 = self.conv2_1(x) 
            x_2 = self.conv2_2(x)
            x = torch.cat([x_0, x_1, x_2], 1)
            x_0 = self.conv3_0(x) 
            x_1 = self.conv3_1(x) 
            x_2 = self.conv3_2(x)
            x = torch.cat([x_0, x_1, x_2], 1)
            x_0 = self.conv4_0(x) 
            x_1 = self.conv4_1(x) 
            x_2 = self.conv4_2(x)
            x = torch.cat([x_0, x_1, x_2], 1)
        else:
            x = self.conv1(x) 
            x = self.conv2(x) 
            x = self.conv3(x)
            if self.non_local:
                x = self.pam(x)
            x = self.conv4(x) 
        x = self.segm(x)
        x = F.softmax(x, dim=1)
        return x
    
    
class DilFCN_PAM(nn.Module):
    def __init__(self):
        super(DilFCN_PAM, self).__init__()
        n_f = 18
        self.conv_init = nn.Sequential(
                                           nn.Conv3d(1, n_f, 3, 1, 1),
                                           nn.InstanceNorm3d(n_f),
                                           nn.ReLU()
                                          )
        self.conv1_0 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv1_1 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv1_2 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv2_0 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv2_1 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv2_2 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv3_0 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv3_1 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv3_2 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv4_0 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 1),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv4_1 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 2, dilation=2),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.conv4_2 = nn.Sequential(
                                       nn.Conv3d(n_f, n_f//3, 3, 1, 4, dilation=4),
                                       nn.InstanceNorm3d(n_f//3),
                                       nn.ReLU()
                                      )
        self.pam = PAM(n_f, n_f//2, n_f//2, 2)
        self.segm = nn.Conv3d(n_f, 25, 1)
        
    def forward(self, x):
        x = self.conv_init(x) 
        x_0 = self.conv1_0(x) 
        x_1 = self.conv1_1(x) 
        x_2 = self.conv1_2(x)
        x = torch.cat([x_0, x_1, x_2], 1)
        x_0 = self.conv2_0(x) 
        x_1 = self.conv2_1(x) 
        x_2 = self.conv2_2(x)
        x = torch.cat([x_0, x_1, x_2], 1)
        x_0 = self.conv3_0(x) 
        x_1 = self.conv3_1(x) 
        x_2 = self.conv3_2(x)
        x = torch.cat([x_0, x_1, x_2], 1)
        x = self.pam(x)
        x_0 = self.conv4_0(x) 
        x_1 = self.conv4_1(x) 
        x_2 = self.conv4_2(x)
        x = torch.cat([x_0, x_1, x_2], 1)
        x = self.segm(x)
        x = F.softmax(x, dim=1)
        return x
    
    
    