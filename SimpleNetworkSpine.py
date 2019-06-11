import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from PAM import PAM

    
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
    
    
    