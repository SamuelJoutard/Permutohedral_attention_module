import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from NetworkSpine import Unet
from DataLoaderHRNSpine import loader, load

def dice_loss(output, gt):
    gt = torch.eye(25).cuda()[gt.type(torch.cuda.LongTensor)]
    gt = gt.permute(0, 4, 1, 2, 3).type(torch.cuda.FloatTensor)
    num = (output*gt).sum(dim=[2, 3, 4])
    denom = output.sum(dim=[2, 3, 4]) + gt.sum(dim=[2, 3, 4]) + 0.001
    res = 1 - (2*num/denom).mean()
    return res

def dice_loss_chill(output, gt):
    gt = torch.eye(25).cuda()[gt.type(torch.cuda.LongTensor)]
    gt = gt.permute(0, 4, 1, 2, 3).type(torch.cuda.FloatTensor)
    num = (output*gt).sum(dim=[2, 3, 4])
    denom = output.sum(dim=[2, 3, 4]) + gt.sum(dim=[2, 3, 4]) + 0.001
    return num, denom


def train(model, n_epoch, DL, DL_test, save_dir, load_=False):
    loss_tracker = []
    loss_tracker_test = []
    if load_:
        loss_tracker = list(np.load(os.path.join(save_dir, "tracker.npy")))
        loss_tracker_test = list(np.load(os.path.join(save_dir, "tracker_test.npy")))
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        print("\nStarting epoch %s ..." % epoch)
        curr_loss = 0
        for i, data in tqdm(enumerate(DL)):
            optimizer.zero_grad()
            inp, lab = data["image"], data["label"]
            inp = inp.type(torch.FloatTensor)
            lab = lab.type(torch.FloatTensor)
            eps = np.random.rand()
            x_ = int(eps*(161-80))
            x__ = x_ + 80
            lab = lab[:, x_:x__, :160, :(inp.size(4)//8)*8]
            inp = inp[:, :, x_:x__, :160, :(inp.size(4)//8)*8]
            inp = inp.cuda()
            lab = lab.cuda()
            out = model(inp)
            loss = criterion(out, lab)
            loss.backward()
            optimizer.step()
            curr_loss += loss.item()
            loss_tracker.append(loss.item())
        print("Training loss: ")
        print(np.array(loss_tracker[-(i+1):]).mean())
        if epoch%10==0:
            test_loss = 0
            model.train(False)
            with torch.no_grad():
                for i, data in enumerate(DL_test):
                    inp, lab = data["image"], data["label"]
                    inp = inp.type(torch.FloatTensor)
                    lab = lab.type(torch.FloatTensor)
                    lab = lab[:, :160, :160, :(inp.size(4)//8)*8]
                    inp = inp[:, :, :160, :160, :(inp.size(4)//8)*8]
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
                    num, denom = dice_loss_chill(out, lab[:, :160])
                    loss = 1 - (2 * num / denom)[:, 1:int(lab.max())+1].mean()
                    test_loss += loss.item()
                loss_tracker_test.append(test_loss/(i+1))
            model.train(True)
            np.save(os.path.join(save_dir, "tracker.npy"), np.array(loss_tracker))
            np.save(os.path.join(save_dir, "tracker_test.npy"), np.array(loss_tracker_test))
        if (epoch%10==0) and (epoch>0):
            torch.save(model.state_dict(), os.path.join(save_dir, "model_weight.pt"))
    np.save(os.path.join(save_dir, "tracker.npy"), np.array(loss_tracker))
    np.save(os.path.join(save_dir, "tracker_test.npy"), np.array(loss_tracker_test))
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weight.pt"))
    return None

    
if __name__=="__main__":
    
    bs = 1
    path_data = "SpineWeb/SpineWeb_15_downsampled"
    
    for k in range(5):
        
        data, label, affine = load(path_data)
        N = len(data)
        data_test = data[4*k:4*(k+1)]
        label_test = label[4*k:4*(k+1)]
        affine_test = affine[4*k:4*(k+1)]
        data = data[:4*k] + data[4*(k+1):]
        label = label[:4*k] + label[4*(k+1):]
        affine = affine[:4*k] + affine[4*(k+1):]
        dataloader = loader(data, label, affine)
        dataloader_test = loader(data_test, label_test, affine_test)
        DL = DataLoader(dataloader, batch_size=bs, shuffle=True)
        DL_test = DataLoader(dataloader_test, batch_size=1, shuffle=False)
        
        load_ = False
        non_local = True
        n_epoch = 50
        save_dir = "3D_Tests/final_results/UnetSpine_nl_bis_relu_fold" + str(k+1)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        model = Unet(non_local=non_local)
        if load_:
            model.load_state_dict(torch.load(os.path.join(save_dir, "model_weight.pt")))
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
        train(model, n_epoch, DL, DL_test, save_dir, load_=load_)
        del(model)
        
        load_ = False
        non_local = False
        n_epoch = 50
        save_dir = "3D_Tests/final_results/UnetSpine_bis_fold" + str(k+1)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        model = Unet(non_local=non_local)
        if load_:
            model.load_state_dict(torch.load(os.path.join(save_dir, "model_weight.pt")))
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
        train(model, n_epoch, DL, DL_test, save_dir, load_=load_)
        del(model)
    

