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
  #def __init__(self , batch_size, retain_dim=False): 
    super().__init__()

    

    #self.operation = config.operation
    self.operation = "concat"

    # create convolution layer before encoder 
    self.PreConv = PreConv(1, 8, kernel_size=(3, 3, 3))

    # create encoder for the model  
    self.encoder1 = Encoder(8,16 ,kernel_size=(7,7,7), same_size = False)
    self.encoder1a = Encoder(16,16 ,kernel_size=(9,9,9),same_size = True)
    self.encoder2 = Encoder(16,32,kernel_size=(5,5,5),same_size = False)
    self.encoder2a = Encoder(32,32,kernel_size=(5,5,5),same_size = True)
    self.encoder3 = Encoder(32,64,kernel_size=(5,5,5),same_size = False)
    self.encoder3a = Encoder(64,64,kernel_size=(7,7,7),same_size = True)
    self.encoder4 = Encoder(64,128, kernel_size=(3,3,3),same_size = False)
    self.encoder4a = Encoder(128,128, kernel_size=(3,3,3),same_size = True)
    self.encoder5 = Encoder(128,256, kernel_size=(3,3,3),same_size = False) 
    self.encoder5a = Encoder(256,256, kernel_size=(1,1,1),same_size = True) 
    #self.encoder6 = Encoder(256,512, kernel_size=(3,3,3),same_size = False) 
    #self.encoder6a = Encoder(512,512, kernel_size=(1,1,1),same_size = True) 
    
    # create Linear layer to concatenate input parameters and encoder output with same size
    self.linear1 = nn.Linear(2048, 1024)
    self.linear2 = nn.Linear(10,1024)

    # create batch normalization before bottleneck layer
    self.bn_before = nn.BatchNorm1d(256)
    # self.bn_before = nn.BatchNorm1d(2048)
    # create bottleneck layer in the network
    # 522: input parameters
    # (None,128,4,1,1): Encoder output
    # self.botleneck = Bottleneck(2058,(None,512,4,1,1))
    self.botleneck = Bottleneck(256,256)

    # create decoder for the x-component model 
    #self.decoderx6a = Decoder(512, 512, kernel_size=(3,3,3), unit_scal_fact=True)
    #self.decoderx6 = Decoder(512, 256, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderx5a = Decoder(256, 256, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decoderx5 = Decoder(256, 128, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderx4a = Decoder(128, 128, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decoderx4 = Decoder(128, 64, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderx3a = Decoder(64, 64, kernel_size=(7,7,7), unit_scal_fact=True)
    self.decoderx3 = Decoder(64, 32, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decoderx2a = Decoder(32, 32, kernel_size=(5,5,5), unit_scal_fact=True)
    self.decoderx2 = Decoder(32, 16, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decoderx1a = Decoder(16, 16, kernel_size=(9,9,9), unit_scal_fact=True)
    self.decoderx1 = Decoder(16, 8,  kernel_size=(7,7,7), unit_scal_fact=False)

    '''
    # create decoder for the y-component model 
    self.decodery6a = Decoder(512, 512, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decodery6 = Decoder(512, 256, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decodery5a = Decoder(256, 256, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decodery5 = Decoder(256, 128, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decodery4a = Decoder(128, 128, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decodery4 = Decoder(128, 64, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decodery3a = Decoder(64, 64, kernel_size=(7,7,7), unit_scal_fact=True)
    self.decodery3 = Decoder(64, 32, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decodery2 = Decoder(32, 16, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decodery1a = Decoder(16, 16, kernel_size=(9,9,9), unit_scal_fact=True)
    self.decodery1 = Decoder(16, 8,  kernel_size=(7,7,7), unit_scal_fact=False)

    # create decoder for the z-component model 
    self.decoderz6a = Decoder(512, 512, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decoderz6 = Decoder(512, 256, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderz5a = Decoder(256, 256, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decoderz5 = Decoder(256, 128, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderz4a = Decoder(128, 128, kernel_size=(3,3,3), unit_scal_fact=True)
    self.decoderz4 = Decoder(128, 64, kernel_size=(3,3,3), unit_scal_fact=False)
    self.decoderz3a = Decoder(64, 64, kernel_size=(7,7,7), unit_scal_fact=True)
    self.decoderz3 = Decoder(64, 32, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decoderz2 = Decoder(32, 16, kernel_size=(5,5,5), unit_scal_fact=False)
    self.decoderz1a = Decoder(16, 16, kernel_size=(9,9,9), unit_scal_fact=True)
    self.decoderz1 = Decoder(16, 8,  kernel_size=(7,7,7), unit_scal_fact=False)
    '''

    # create Convolution layer after the decoder 
    self.PostConv_concat = PostConv(5, 1, kernel_size=(3,3, 3))

    self.PostConv = PostConv(9, 1, kernel_size=(3,3, 3))
    #self.PostConv = PostConv(17, 1, kernel_size=(3,3, 3))

    self.retain_dim = retain_dim


  def forward(self, input):

    input = input[:, None, ...]
    # pass the input through pre convolution layer
    x = self.PreConv(input)   #b*1*256*64*64


    # pass the pre convolution output to the enocder

    enc_1 = self.encoder1(x)
    enc_1a = self.encoder1a(enc_1)
    enc_2 = self.encoder2(enc_1a)
    enc_2a = self.encoder2a(enc_2)
    enc_3 = self.encoder3(enc_2a)
    enc_3a = self.encoder3a(enc_3)
    enc_4 = self.encoder4(enc_3a)
    enc_4a = self.encoder4a(enc_4)
    enc_5 = self.encoder5(enc_4a)
    enc_5a = self.encoder5a(enc_5)
    # store the last encoder output shape 
    encoder_output_shape = enc_5a.shape

    # flatten the encoder output
    flaten_array = enc_5a.flatten(start_dim=1)
    # flaten_array = enc_5.flatten()
  
    # flatten the inital parameters
    #param_flat = param.flatten(start_dim=1)
    # param_flat = param.flatten()

    #enc_linear = self.linear1(flaten_array)
    enc_linear = flaten_array
    #param_linear = self.linear2(param_flat)
    
    # concatenate encoder output and intial parameters

    # For GPU 
    #bottleneck_inp = torch.cat([enc_linear, param_linear], 1).cuda()

    # For CPU 
    #bottleneck_inp = torch.cat([enc_linear, param_linear], 1)


    # pass concatenated result in the batch normalization
    bottleneck_inp = self.bn_before(enc_linear)

    # pass batch normalization into bottleneck layer
    bottleneck = self.botleneck(bottleneck_inp)

    # reshape the bottleneck to pass into decoder
    bottleneck = torch.reshape(bottleneck, encoder_output_shape)


    # pass the bottleneck output and encoder output from skip conncection to the x-component decoder
    dec_x5a = self.decoderx5a(bottleneck, enc_5a)
    dec_x5 = self.decoderx5(dec_x5a, enc_5)
    dec_x4a = self.decoderx4a(dec_x5, enc_4a)
    dec_x4 = self.decoderx4(dec_x4a, enc_4)
    dec_x3a = self.decoderx3a(dec_x4, enc_3a)
    dec_x3 = self.decoderx3(dec_x3a, enc_3)
    dec_x2a = self.decoderx2a(dec_x3, enc_2a)
    dec_x2 = self.decoderx2(dec_x2a, enc_2)
    dec_x1a = self.decoderx1a(dec_x2, enc_1a)
    dec_x1 = self.decoderx1(dec_x1a, enc_1)

    '''
    
    # pass the bottleneck output and encoder output from skip conncection to the y-component decoder
    dec_y6a = self.decodery6a(bottleneck, enc_6a)
    dec_y6 = self.decodery6(dec_y6a, enc_6)
    dec_y5a = self.decodery5a(dec_y6, enc_5a)
    dec_y5 = self.decodery5(dec_y5a, enc_5)
    dec_y4a = self.decodery4a(dec_y5, enc_4a)
    dec_y4 = self.decodery4(dec_y4a, enc_4)
    dec_y3a = self.decodery3a(dec_y4, enc_3a)
    dec_y3 = self.decodery3(dec_y3a, enc_3)
    dec_y2 = self.decodery2(dec_y3, enc_2)
    dec_y1a = self.decodery1a(dec_y2, enc_1a)
    dec_y1 = self.decodery1(dec_y1a, enc_1)


    # pass the bottleneck output and encoder output from skip conncection to the z-component decoder
    dec_z6a = self.decoderz6a(bottleneck, enc_6a)
    dec_z6 = self.decoderz6(dec_z6a, enc_6)
    dec_z5a = self.decoderz5a(dec_z6, enc_5a)
    dec_z5 = self.decoderz5(dec_z5a, enc_5)
    dec_z4a = self.decoderz4a(dec_z5, enc_4a)
    dec_z4 = self.decoderz4(dec_z4a, enc_4)
    dec_z3a = self.decoderz3a(dec_z4, enc_3a)
    dec_z3 = self.decoderz3(dec_z3a, enc_3)
    dec_z2 = self.decoderz2(dec_z3, enc_2)
    dec_z1a = self.decoderz1a(dec_z2, enc_1a)
    dec_z1 = self.decoderz1(dec_z1a, enc_1)
    '''


    dec_x  = torch.cat((dec_x1, input), 1)
    #dec_y   = torch.cat((dec_y1, input), 1)
    #dec_z  = torch.cat((dec_z1, input), 1)   

    out_x = self.PostConv(dec_x)
    #out_y = self.PostConv(dec_y)
    #out_z = self.PostConv(dec_z)


    #out = torch.cat((out_x, out_y, out_z), 1)

    return out_x