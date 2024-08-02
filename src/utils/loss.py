import numpy as np
import torch

def nrmse_loss(x, y, p = 2, w=None, reduction=True):
    err =  torch.abs(x-y)**p
    norm_y = torch.abs(y)**p
    dims_sum = tuple(range(2,len(x.shape)))

    err =  torch.sum(err, dim=dims_sum)
    norm_y =  torch.sum(norm_y, dim=dims_sum)

    epsilon = 1e-8

    loss = err/(norm_y + epsilon)

    if w:
        loss = w * loss

    if reduction:   
        loss = torch.mean(loss)
    else:
        loss = torch.mean(loss, dim=tuple(range(1,len(err.shape))))
    return loss

def rH1loss(x, y, w=None, reduction=True):
    grad_x = torch.cat(torch.gradient(x, dim=tuple(range(2,len(x.shape)-1))), 1)
    grad_y = torch.cat(torch.gradient(y, dim=tuple(range(2,len(x.shape)-1))), 1)
    
    return nrmse_loss(grad_x, grad_y, w=w, reduction=reduction)

def spec_loss(x, y, w=None, reduction=True):
    y_hat = torch.fft.rfftn(y, dim=tuple(range(2,len(x.shape)-1)))
    x_hat = torch.fft.rfftn(x, dim=tuple(range(2,len(y.shape)-1)))
    
    return nrmse_loss(x_hat, y_hat, w=w, reduction=reduction)