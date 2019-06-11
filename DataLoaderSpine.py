from torch.utils.data import Dataset, DataLoader
import torch
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm

def load(list_dir):
    data = []
    label = []
    affine = []
    for i, fold in enumerate([list_dir]):
        for fold_ in tqdm(os.listdir(fold)):
            if "sub" in fold_:
                im_file = os.path.join(fold, fold_) + "/" + fold_ + "_CT.nii.gz"
                label_file = os.path.join(fold, fold_) + "/" + fold_ + "_segmentation_relab_bis.nii.gz"
                im = nib.load(im_file,)
                lab = nib.load(label_file, )
                affine_im = im.affine
                im = im.get_fdata().astype(np.float32)
                lab = lab.get_fdata().astype(np.float32)
                im = im
                if binary:
                    lab[lab>0] = 1
                lab = lab.astype(int)
                data.append(im[np.newaxis])
                label.append(lab)
                affine.append(affine_im)
    
    return data, label, affine


class loader(Dataset):
    
    def __init__(self, im, label, affine):
        super(loader, self).__init__()
        self.data, self.label, self.affine = im, label, affine
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        im = self.data[idx]
        lab = self.label[idx]
        sample = {"image": im,
                  "label": lab,
                  "affine": self.affine[idx]}
        return sample