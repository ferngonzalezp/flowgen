import torch 
import torch.nn as nn
import torchvision
from model.block import*
from model.DecoderBlock import*
from model.encoder import Encoder
from model.bottleneck import*

class UNet(nn.Module):

  """
  Class to implement model 
  """

  def __init__(self,  retain_dim=False): 
    super().__init__()

    # self.operation = config.operation

    # create convolution layer before encoder 
    self.PreConv = PreConv(1, 8, kernel_size=(3, 3, 3))

    # create encoder for the model  
    self.encoder1 = Encoder(8,16 ,kernel_size=(7,7,7))
    self.encoder2 = Encoder(16,32,kernel_size=(5,5,5))
    self.encoder3 = Encoder(32,64,kernel_size=(5,5,5))
    self.encoder4 = Encoder(64,128, kernel_size=(3,3,3))
    self.encoder5 = Encoder(128,256, kernel_size=(3,3,3)) 
    
    # create Linear layer to concatenate input parameters and encoder output with same size
    self.linear1 = nn.Linear(8192, 4096)
    # self.linear1b = nn.Linear(4096, 256)

    self.linear2 = nn.Linear(10,4096)
    # self.linear2 = nn.Linear(4208,256)

    # create batch normalization before bottleneck layer
    self.bn_before = nn.BatchNorm1d(8192)

    self.botleneck = Bottleneck(8192,8192)

    # create decoder for the x-component model 
    self.decoderx1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decoderx2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decoderx3 = Decoder(64, 32, kernel_size=(5,5,5))
    self.decoderx4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decoderx5 = Decoder(16, 8,  kernel_size=(7,7,7))

    # create decoder for the y-component model 
    self.decodery1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decodery2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decodery3 = Decoder(64, 32, kernel_size=(5,5,5))
    self.decodery4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decodery5 = Decoder(16, 8,  kernel_size=(7,7,7))

    # create decoder for the z-component model 
    self.decoderz1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decoderz2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decoderz3 = Decoder(64, 32, kernel_size=(5,5,5))
    self.decoderz4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decoderz5 = Decoder(16, 8,  kernel_size=(7,7,7))


    self.PostConv = PostConv(9, 1, kernel_size=(3,3, 3))

   
    self.retain_dim = retain_dim

  def forward(self, in_put, param):
    
    # print("input shape ", input.shape)
    # print("param shape ", param.shape)

    in_put = in_put[:, None, ...] # b*1*256*64*64

    # pass the input through pre convolution layer
    x = self.PreConv(in_put)
    # pass the pre convolution output to the enocder
    enc_1 = self.encoder1(x)
    enc_2 = self.encoder2(enc_1)
    enc_3 = self.encoder3(enc_2)
    enc_4 = self.encoder4(enc_3)
    enc_5 = self.encoder5(enc_4)
    # print('enc_4: ',enc_4.shape)

    # print('enc_5: ',enc_5.shape)
    # store the last encoder output shape 
    encoder_output_shape = enc_5.shape

    # flatten the encoder output
    flaten_array = enc_5.flatten(start_dim=1)   # b*8192 = b * 256*8*2*2
    # flaten_array = enc_5.flatten()

    # flatten the inital parameters
    param_flat = param.flatten(start_dim=1)     #b*10
    # param_flat = param.flatten()
    
    enc_linear = self.linear1(flaten_array)

    param_linear = self.linear2(param_flat)
    
    # concatenate encoder output and intial parameters
    # bottleneck_inp = torch.cat([flaten_array, param_flat], 1).cuda()
    #bottleneck_inp = torch.cat([enc_linear, param_linear], 1).cuda()
    bottleneck_inp = torch.cat([enc_linear, param_linear], 1)

    # pass concatenated result in the batch normalization
    # bottleneck_inp = self.bn_before(bottleneck_inp)
    # print('bn1: ', bottleneck_inp.shape)

    # pass batch normalization into bottleneck layer
    bottleneck = self.botleneck(bottleneck_inp)
    # reshape the bottleneck to pass into decoder
    bottleneck = torch.reshape(bottleneck, encoder_output_shape)

    # pass the bottleneck output and encoder output from skip conncection to the x-component decoder
    dec_x1 = self.decoderx1(bottleneck, enc_5)
    dec_x2 = self.decoderx2(dec_x1, enc_4)
    dec_x3 = self.decoderx3(dec_x2, enc_3)
    dec_x4 = self.decoderx4(dec_x3, enc_2)
    dec_x5 = self.decoderx5(dec_x4, enc_1)

    # pass the bottleneck output and encoder output from skip conncection to the y-component decoder
    dec_y1 = self.decodery1(bottleneck, enc_5)
    dec_y2 = self.decodery2(dec_y1, enc_4)
    dec_y3 = self.decodery3(dec_y2, enc_3)
    dec_y4 = self.decodery4(dec_y3, enc_2)
    dec_y5 = self.decodery5(dec_y4, enc_1)

    # pass the bottleneck output and encoder output from skip conncection to the z-component decoder
    dec_z1 = self.decoderz1(bottleneck, enc_5)
    dec_z2 = self.decoderz2(dec_z1, enc_4)
    dec_z3 = self.decoderz3(dec_z2, enc_3)
    dec_z4 = self.decoderz4(dec_z3, enc_2)
    dec_z5 = self.decoderz5(dec_z4, enc_1)


    # if self.operation == "concat":
    dec_x = torch.cat((dec_x5, in_put), 1)
    dec_y = torch.cat((dec_y5, in_put), 1)
    dec_z = torch.cat((dec_z5, in_put), 1)   

    out_x = self.PostConv(dec_x)
    out_y = self.PostConv(dec_y)
    out_z = self.PostConv(dec_z)

    out = torch.cat((out_x, out_y, out_z), 1)


    return out