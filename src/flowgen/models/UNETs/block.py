import torch 
import torch.nn as nn
import torchvision
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from einops import rearrange, einsum
from functools import partial
import math

class Conv3dSame(torch.nn.Conv3d):
    
    """
    This class perform convolution by calculating same padding for different kernel size and 
    returns the same kernel size after convolution

    Returns:
    """
    
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
    
        """
        This function claculates padding 
        Args:
            i: size of the input tensor
            k: kernel size
            s: stride
            d: dilation [not significant here]
    
        Returns:
        """
    
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw, ide = x.size()[-3:]

        # calulates padding for height width and depth of the tensor
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=ide, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2]
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )



class PreConv(nn.Module):
    
    """
    This class creates pre convolution layer in the network
    Args:
        inchs: number of input channels
        outchs: number of output channels
        kernel_size: kernel size
        x: input to the Preconvolution 
    
    Returns: 
    """
    

    def __init__(self, inchs, outchs, kernel_size):
        super().__init__()
        
        # pading of the input tensor boundaries with a constant value
        conv_padding = reduce(__add__, 
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ConstantPad3d(conv_padding,value=0)       
        self.Conv1 = nn.Conv3d(inchs, outchs, kernel_size=kernel_size)


    def forward(self, x):
        
        # introduce an extra axis
        
        inp = self.pad(x)       #b*1*258*66*66
        out = self.Conv1(inp)   #b*8*256*64*64
               
        return out     


'''
class squeeze_and_excite_encoder(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        self.channels =  channels

        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid())

    def forward(self, x):
        bs, c, h, w, d = x.shape
        y = self.avg_pool(x).view(bs, self.channels)
        
        y = self.fc(y).view(bs, self.channels, 1, 1, 1)

        return x * y.expand_as(x)

'''
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()

        self.channels =  channels
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)

        channel_att = torch.sigmoid(self.fc(avg_pool).view(x.size(0), x.size(1), 1, 1, 1))

        return channel_att


class SpatialAttention(nn.Module):

    def __init__(self, in_ch,out_ch, kernel_size):

        super(SpatialAttention, self).__init__()

        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        

        #self.conv1 = nn.Conv3d(2, 1, kernel_size_1, padding=kernel_size_1 // 2, bias=False)


    def forward(self, x):

        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        pool = torch.cat([max_pool, avg_pool], dim=1)

        spatial_att = torch.sigmoid(self.conv1(pool))

        return spatial_att

class Block(nn.Module):
    
    """
    This class consists of block of the Encoder Block which includes three convolution layer. 
    Note: Conv3dSame returns same kernel size 
    Args:
        in_ch: number of input channels
        out_ch: number of output channels
        kernel_size: kernel size
        x: input to the encoder block
        out: output of the encoder block

    Returns: 
    """
  
    def __init__(self, in_ch, out_ch, kernel_size, same_size):
        super().__init__()

        self.activation2 = nn.LeakyReLU(0.2)
        self.conv2 = Conv3dSame(out_ch, out_ch, kernel_size)
        self.batchnorm2 = nn.BatchNorm3d(out_ch)
        
        '''
        if same_size:  
            conv_padding = tuple((k - 1) // 2 for k in kernel_size)
        else:
            conv_padding = reduce(__add__, 
                                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        '''

        #self.se_encoder = squeeze_and_excite_encoder(in_ch)
        self.channel_attention = ChannelAttention(in_ch)
        self.spatial_attention = SpatialAttention(in_ch,out_ch,kernel_size)

        conv_padding = reduce(__add__, 
                    [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
            # self.pad = nn.ConstantPad3d(conv_padding,value=0) 
    
        self.pad = nn.ReflectionPad2d(conv_padding)


        if same_size:  

            self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        else:

            self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=2)

        self.batchnorm = nn.BatchNorm3d(out_ch)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout3d(p=0.01)

        if same_size: 

            self.residualConv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            
            self.residualConv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=2)


    def forward(self, x):

        #NOTE: Try to use only either se_encoder or channel_att, as both of them are almost similar

        #x_se_encoder  = self.se_encoder(x)
        x_se_encoder = x

        channel_att = self.channel_attention(x_se_encoder)
        x_se_encoder = x_se_encoder * channel_att

        spatial_att = self.spatial_attention(x_se_encoder)
        x_se_encoder = x_se_encoder * spatial_att

        res = x_se_encoder             #b*8*256*64*64
        out = self.pad(x_se_encoder)   #b*8*262*70*70
        
        out = self.activation(out)  #b*8*262*70*70
        out = self.conv3(out)       #b*16*128*32*32
        out = self.batchnorm(out)   #b*16*128*32*32
        

        out = self.dropout(out)     #b*16*128*32*32
        # print('res: ', res.shape)

        residual = self.residualConv(res)   #b*16*128*32*32
        # print("conres: ", residual.shape)

        # print('out: ', out.shape)
        out = out + residual                #b*16*128*32*32

        # outb = self.activation2(out)
        # outb = self.conv2(outb)
        # outb = self.batchnorm2(outb)
        # outb = self.dropoutb(outb)

        return out

#@partial(torch.compile)
class ConvNextBlockBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, D, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6, out_dim=None):
        super().__init__()
        if not out_dim:
            out_dim = dim
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, out_dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.residualConv = nn.Conv3d(dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        res = self.residualConv(x)
        out = self.dwconv(x)
        #x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        #x = rearrange(x, 'b c h w d -> b h w d c').contiguous()
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = einsum(out, self.gamma, 'b c h w d, c -> b c h w d')
            #x = self.gamma * x
        #x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W, D)
        #x = rearrange(x, 'b h w d c -> b c h w d').contiguous()
        out += res
        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        B, C, *dim = x.shape
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.reshape(B, C, -1)
            x = self.weight[None, :, None] * x + self.bias[None, :, None] 
            return x.reshape(B, C, *dim)
