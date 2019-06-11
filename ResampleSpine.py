import nibabel as nib
import os
from nibabel.processing import resample_to_output
from tqdm import tqdm
import numpy as np

path_bis = "SpineWeb/SpineWeb_15_downsampled"
if not os.path.isdir(path_bis):
    os.mkdir(path_bis)

path_img = 'SpineWeb/SpineWeb_15'


for folder in tqdm(os.listdir(path_img)):
    if "sub" in folder:
        im_file = os.path.join(path_img, folder) + "/" + folder + "_CT.nii.gz"
        label_file = os.path.join(path_img, folder) + "/" + folder + "_segmentation.nii.gz"
        im = nib.load(im_file)
        seg = nib.load(label_file)

        im = resample_to_output(im,(1,1,3), order=3)
        seg = resample_to_output(seg,(1,1,3), order=0)

        affine = seg.affine
        un = np.unique(seg)
        for i in range(1, len(un)):
            seg[seg==un[i]] = len(un) - i 

        if folder=="sub-013":
            print("resampling 13")
            seg[seg<4] = 0
            seg[seg>0] = seg[seg>0] - 3
        if folder=="sub-011":
            seg[seg<4] = 0
            seg[seg>0] = seg[seg>0] - 3
            seg[seg==19] = 0
        if folder=="sub-004":
            seg[seg<2] = 0
            seg[seg>0] = seg[seg>0] - 1
        if folder=="sub-012":
            seg[seg<6] = 0
            seg[seg>0] = seg[seg>0] - 5
        if folder=="sub-015":
            seg[seg<5] = 0
            seg[seg>0] = seg[seg>0] - 4
            
        seg = nib.Nifti1Image(seg.astype(float), affine=affine)
        
        new_path_im = os.path.join(path_bis, folder)
        if not os.path.isdir(new_path_im):
            os.mkdir(new_path_im)
            
        im_name = folder + "_CT.nii.gz"
        lab_name = folder + "_segmentation.nii.gz"

        nib.save(im, os.path.join(new_path_im, im_name))
        nib.save(seg, os.path.join(new_path_im, lab_name))
