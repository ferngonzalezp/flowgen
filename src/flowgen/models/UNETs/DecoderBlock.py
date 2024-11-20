import torch 
import torch.nn as nn
import torchvision
from model.block import*
from functools import reduce
from operator import __add__

class Upsample(nn.Module):
    
    """
    This Class does upsampling in the decoder in the network
    
    Args:
        in_channels: number of input channels
        out_channels: number of output channels

    Returns: 
    """
    
    def __init__(self, in_channels, out_channels, unit_scal_fact, kernel_size):

        super(Upsample, self).__init__()

        if unit_scal_fact:
            
            self.up_block = nn.Upsample(scale_factor=1, mode='trilinear') 
        
        else:
            self.up_block = nn.Upsample(scale_factor=2, mode='trilinear') #TODO:
           
            
        self.deconv1 = Conv3dSame(in_channels, out_channels, kernel_size=kernel_size)

        self.batchnorm1 = nn.BatchNorm3d(out_channels)

        self.dropout = nn.Dropout3d(p=0.01)
            
        self.activation1 = nn.ReLU()


    def forward(self, x):
        out = self.up_block(x)
        out = self.activation1(out)
        out = self.deconv1(out)
        out = self.batchnorm1(out)
       
        
        # out = self.dropout(out)

        return out



class Decoderblock(nn.Module):
    
    """
    This class is to create Decoder block in the network
    
    Args:
        in_ch: number of input channels
        out_ch: number of output channels
        kernel_size: kernel size
        x: input to decoder block [upsampling output] 

    Returns:
    """
    
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()


        conv_padding = reduce(__add__, 
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        # self.pad = nn.ConstantPad3d(conv_padding,value=0)    
        self.pad = nn.ReflectionPad2d(conv_padding)
        self.deconv3 = nn.Conv3d(in_ch, out_ch, kernel_size)
        self.batchnorm = nn.BatchNorm3d(out_ch)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.01)



    def forward(self, x):
        out = self.pad(x)
        out = self.activation(out)
        out = self.deconv3(out)
        out = self.batchnorm(out)
        # out = self.dropout(out)
        
        return out

'''
class squeeze_and_excite(nn.Module):
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
    
class Decoder(nn.Module):
    
    """
    This class is to create decoder in the network which includes first uosampling and follows by Decoder block
    
    Args:
        in_chs: number of input channels
        out_chs: number of output channels
        kernel_size: kernel size
        x: input to decoder from bottleneck layer or previous decoder block
        encoder_out: output from encoder [skip connection]

    Returns:
    """
    
    def __init__(self, in_chs, out_chs, unit_scal_fact, kernel_size):
        super().__init__()

        #self.se = squeeze_and_excite(in_chs)
        self.channel_attention = ChannelAttention(in_chs)
        self.spatial_attention = SpatialAttention(in_chs,out_chs,kernel_size)

        self.upconvs = Upsample(2*in_chs, out_chs, unit_scal_fact, kernel_size=kernel_size)

        # self.residualConv = nn.ConvTranspose3d(in_chs, out_chs, kernel_size=1, stride=2, padding=(1,1,1))
        
        # self.dec_blocks = Decoderblock(2*out_chs, out_chs, kernel_size)



    def forward(self, dec_in, encoder_out):

        #se1 = self.se(dec_in)
        se1 = dec_in


        channel_att_dc_1 = self.channel_attention(se1)
        se1 = se1 * channel_att_dc_1

        spatial_att_dc_1 = self.spatial_attention(se1)
        se1 = se1 * spatial_att_dc_1

        #se2 = self.se(encoder_out)
        se2 = encoder_out

        channel_att_dc_2 = self.channel_attention(se2)
        se2 = se2 * channel_att_dc_2


        spatial_att_dc_2 = self.spatial_attention(se2)
        se2 = se2 * spatial_att_dc_2


        # concatinating output from previos block and skip connections 
        x = torch.cat([se1, se2], dim=1)
        x = self.upconvs(x)

        # x = self.dec_blocks(x)
       
        return x


class PostConv(nn.Module):
    
    """
    This Class is to create Post convolution at the end of the network
    
    Args: 
        inchs: number of input channels
        outchs: number of output channels
        kernel_size = kernel size
        x: input to the PostConv from last decoder 
        out: output from the decoder
    
    Returns:
    """
    
    def __init__(self, inchs=6, outchs=1, kernel_size=(3,3)):
        super().__init__()
        conv_padding = reduce(__add__, 
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        
        self.pad = nn.ConstantPad3d(conv_padding, value=0)
        
        self.Conv1 = nn.Conv3d(inchs, outchs, kernel_size=kernel_size)
        
       

    def forward(self, x):
        
        # Calculate padding and pass the output to convolution layer
        inp = self.pad(x)
        out = self.Conv1(inp)
        
        return out     
