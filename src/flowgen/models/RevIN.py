import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.mean = []
        self.stdev = []
        if self.affine:
            self._init_params()

    def forward(self, x, index, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x, index)
        elif mode == 'denorm':
            x = self._denormalize(x, index)
        else: raise NotImplementedError
        return x
    
    def reinit_stats(self):
        self.mean = []
        self.stdev = []

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight_in = nn.Sequential(nn.Conv1d(in_channels=self.num_features, out_channels=128, kernel_size=1, bias=False),
                                           nn.GELU(),
                                           nn.Conv1d(in_channels=128, out_channels=self.num_features, kernel_size=1, bias=False))
        self.affine_bias_in = nn.Sequential(nn.Conv1d(in_channels=self.num_features, out_channels=128, kernel_size=1, bias=False),
                                           nn.GELU(),
                                           nn.Conv1d(in_channels=128, out_channels=self.num_features, kernel_size=1, bias=False))
        self.affine_weight_out = nn.Sequential(nn.Conv1d(in_channels=self.num_features, out_channels=128, kernel_size=1, bias=False),
                                           nn.GELU(),
                                           nn.Conv1d(in_channels=128, out_channels=self.num_features, kernel_size=1, bias=False))
        self.affine_bias_out = nn.Sequential(nn.Conv1d(in_channels=self.num_features, out_channels=128, kernel_size=1, bias=False),
                                           nn.GELU(),
                                           nn.Conv1d(in_channels=128, out_channels=self.num_features, kernel_size=1, bias=False))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(2, x.ndim))
        self.mean.append(torch.mean(x, dim=dim2reduce, keepdim=True).detach())
        self.stdev.append(torch.std(x, dim=dim2reduce, keepdim=True, unbiased=False).detach())

    def _normalize(self, x, index):
        bs, c, *dim = x.shape
        dim2reduce = tuple(range(2, x.ndim))
        #Standarize
        x = x - self.mean[index]
        x = x / (self.stdev[index] + self.eps)

        if self.affine:
            gamma = self.affine_weight_in(x.reshape(bs,c,-1)).reshape(bs,c,*dim)
            beta =  self.affine_bias_in(x.reshape(bs,c,-1)).reshape(bs,c,*dim)
            x = torch.einsum('bf... , bf... -> bf...', x, gamma)
            x = x + beta
        return x

    def _denormalize(self, x, index):
        bs, c, *dim = x.shape
        if self.affine:
            gamma = self.affine_weight_out(x.reshape(bs,c,-1)).reshape(bs,c,*dim)
            beta =  self.affine_bias_out(x.reshape(bs,c,-1)).reshape(bs,c,*dim)
            x = torch.einsum('bf... , bf... -> bf...', x, gamma)
            x = x + beta
        
        x = x * self.stdev[index]
        x = x + self.mean[index]
        return x
    