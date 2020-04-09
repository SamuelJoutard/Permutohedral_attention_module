import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from NetworkSpine import HighResNet, Unet
from Unet_asym import Unet as Unet_asym
from DataLoaderSpine import loader, load




def visualize(model, save_dir, DL):
    for i, data in tqdm(enumerate(DL)):
        inp, lab, affine= data["image"], data["label"], data["affine"]
        lab = lab[:, :160, :160, :(inp.size(4)//8)*8]
        inp = inp[:, :, :160, :160, :(inp.size(4)//8)*8]
        inp_save = nib.Nifti1Image(inp.numpy()[0, 0], affine=affine[0])
        lab_save = nib.Nifti1Image(lab.numpy()[0].astype(float), affine=affine[0])
        nib.save(inp_save, os.path.join(save_dir, "im_"+str(i)+".nii.gz"))
        nib.save(lab_save, os.path.join(save_dir, "gt_"+str(i)+".nii.gz"))
        inp = inp.type(torch.FloatTensor)
        lab = lab.type(torch.FloatTensor)
        with torch.no_grad():
            inp_ = inp[:, :, :80].cuda()
            out_1 = model(inp_)
            inp_ = inp[:, :, 40:120].cuda()
            out_2 = model(inp_)
            inp_ = inp[:, :, 80:160].cuda()
            out_3 = model(inp_)
            out = torch.cat([out_1[:, :, :40],
                            0.5 * out_1[:, :, 40:80] + 0.5 * out_2[:, :, 0:40],
                            0.5 * out_2[:, :, 40:80] + 0.5 * out_3[:, :, 0:40],
                            out_3[:, :, 40:]], 2)
            out = out.argmax(dim=1)
        out = out.cpu().numpy()[0]
        out = nib.Nifti1Image(out.astype(float), affine=affine[0])
        nib.save(out, os.path.join(save_dir, "seg_"+str(i)+".nii.gz"))


if __name__=="__main__":
    path_data = "SpineWeb/SpineWeb_15_downsampled"
    data, label, affine = load(path_data)
    dataloader = loader(data, label, affine)
    DL = DataLoader(dataloader, batch_size=1, shuffle=False)
    
    for k in range(5):
        non_local = False
        save_dir = "3D_Tests/final_results/UnetSpine_bis_fold" + str(k+1)
        model = Unet(non_local=non_local)
        model.load_state_dict(torch.load(os.path.join(save_dir, "model_weight.pt")))
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
        visualize(model, save_dir, DL)
        del(model)
        
    for k in range(5):
        non_local = True
        save_dir = "3D_Tests/final_results/UnetSpine_nl_bis_relu_fold" + str(k+1)
        model = Unet(non_local=non_local)
        model.load_state_dict(torch.load(os.path.join(save_dir, "model_weight.pt")))
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
        visualize(model, save_dir, DL)
        del(model)
        