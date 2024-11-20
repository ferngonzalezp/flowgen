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
    self.PreConv = PreConv(1, 4, kernel_size=(3, 3, 3))

    # create encoder for the model  
    self.encoder1 = Encoder(4,8 ,kernel_size=(7,7,7))
    self.encoder2 = Encoder(8,16,kernel_size=(5,5,5))
    self.encoder3 = Encoder(16,32,kernel_size=(5,5,5))
    self.encoder4 = Encoder(32,64, kernel_size=(3,3,3))
    self.encoder5 = Encoder(64,128, kernel_size=(3,3,3)) 
    self.encoder6 = Encoder(128,256, kernel_size=(3,3,3)) 
    
    # create Linear layer to concatenate input parameters and encoder output with same size
    self.linear1 = nn.Linear(1024, 512)
    self.linear2 = nn.Linear(10,512)

    # create batch normalization before bottleneck layer
    self.bn_before = nn.BatchNorm1d(1024)
    # self.bn_before = nn.BatchNorm1d(2048)
    # create bottleneck layer in the network
    # 522: input parameters
    # (None,128,4,1,1): Encoder output
    # self.botleneck = Bottleneck(2058,(None,512,4,1,1))
    self.botleneck = Bottleneck(1024,1024)

    # create decoder for the x-component model 
    self.decoderx1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decoderx2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decoderx3 = Decoder(64, 32, kernel_size=(3,3,3))
    self.decoderx4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decoderx5 = Decoder(16, 8, kernel_size=(5,5,5))
    self.decoderx6 = Decoder(8, 4,  kernel_size=(7,7,7))

    # create decoder for the y-component model 
    self.decodery1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decodery2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decodery3 = Decoder(64, 32, kernel_size=(3,3,3))
    self.decodery4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decodery5 = Decoder(16, 8, kernel_size=(5,5,5))
    self.decodery6 = Decoder(8, 4,  kernel_size=(7,7,7))

    # create decoder for the z-component model 
    self.decoderz1 = Decoder(256, 128, kernel_size=(3,3,3))
    self.decoderz2 = Decoder(128, 64, kernel_size=(3,3,3))
    self.decoderz3 = Decoder(64, 32, kernel_size=(3,3,3))
    self.decoderz4 = Decoder(32, 16, kernel_size=(5,5,5))
    self.decoderz5 = Decoder(16, 8, kernel_size=(5,5,5))
    self.decoderz6 = Decoder(8, 4,  kernel_size=(7,7,7))

    # self.conlayer1 = nn.Linear(5,4,(1,1,1))
    # self.conlayer2 = nn.Linear(5,4,(1,1,1))
    # self.conlayer3 = nn.Linear(5,4,(1,1,1))

    # create Convolution layer after the decoder 
    self.PostConv_concat = PostConv(5, 1, kernel_size=(3,3, 3))

    self.PostConv = PostConv(4, 1, kernel_size=(3,3, 3))

   
    self.retain_dim = retain_dim

  def forward(self, input, param):


    input = input[:, None, ...]
    # pass the input through pre convolution layer
    x = self.PreConv(input)   #b*1*256*64*64

    # pass the pre convolution output to the enocder
    enc_1 = self.encoder1(x)
    enc_2 = self.encoder2(enc_1)
    enc_3 = self.encoder3(enc_2)
    enc_4 = self.encoder4(enc_3)
    enc_5 = self.encoder5(enc_4)
    enc_6 = self.encoder6(enc_5)
    # store the last encoder output shape 
    encoder_output_shape = enc_6.shape

    # flatten the encoder output
    flaten_array = enc_6.flatten(start_dim=1)
    # flaten_array = enc_5.flatten()
  
    # flatten the inital parameters
    param_flat = param.flatten(start_dim=1)
    # param_flat = param.flatten()

    enc_linear = self.linear1(flaten_array)
    param_linear = self.linear2(param_flat)
    
    # concatenate encoder output and intial parameters

    # For GPU 
    #bottleneck_inp = torch.cat([enc_linear, param_linear], 1).cuda()

    # For CPU 
    bottleneck_inp = torch.cat([enc_linear, param_linear], 1)


    # pass concatenated result in the batch normalization
    bottleneck_inp = self.bn_before(bottleneck_inp)

    # pass batch normalization into bottleneck layer
    bottleneck = self.botleneck(bottleneck_inp)

    # reshape the bottleneck to pass into decoder
    bottleneck = torch.reshape(bottleneck, encoder_output_shape)

    # pass the bottleneck output and encoder output from skip conncection to the x-component decoder
    dec_x1 = self.decoderx1(bottleneck, enc_6)
    dec_x2 = self.decoderx2(dec_x1, enc_5)
    dec_x3 = self.decoderx3(dec_x2, enc_4)
    dec_x4 = self.decoderx4(dec_x3, enc_3)
    dec_x5 = self.decoderx5(dec_x4, enc_2)
    dec_x6 = self.decoderx6(dec_x5, enc_1)

    # pass the bottleneck output and encoder output from skip conncection to the y-component decoder
    dec_y1 = self.decodery1(bottleneck, enc_6)
    dec_y2 = self.decodery2(dec_y1, enc_5)
    dec_y3 = self.decodery3(dec_y2, enc_4)
    dec_y4 = self.decodery4(dec_y3, enc_3)
    dec_y5 = self.decodery5(dec_y4, enc_2)
    dec_y6 = self.decodery6(dec_y5, enc_1)

    # pass the bottleneck output and encoder output from skip conncection to the z-component decoder
    dec_z1 = self.decoderz1(bottleneck, enc_6)
    dec_z2 = self.decoderz2(dec_z1, enc_5)
    dec_z3 = self.decoderz3(dec_z2, enc_4)
    dec_z4 = self.decoderz4(dec_z3, enc_3)
    dec_z5 = self.decoderz5(dec_z4, enc_2)
    dec_z6 = self.decoderz6(dec_z5, enc_1)

    # y = torch.cat((dec_5, x), 1)
    # add input to the decoder output
    # dec_x5 = dec_x5 + x
    # dec_y5 = dec_y5 + x
    # dec_z5 = dec_z5 + x

    if self.operation == "concat":
      dec_x6 = torch.cat((dec_x6, input), 1)
      dec_y6 = torch.cat((dec_y6, input), 1)
      dec_z6 = torch.cat((dec_z6, input), 1)   

      out_x = self.PostConv_concat(dec_x6)
      out_y = self.PostConv_concat(dec_y6)
      out_z = self.PostConv_concat(dec_z6)

    
    elif self.operation == "add":
      dec_x6 = dec_x6 + input
      dec_y6 = dec_y6 + input
      dec_z6 = dec_z6 + input

      out_x = self.PostConv(dec_x6)
      out_y = self.PostConv(dec_y6)
      out_z = self.PostConv(dec_z6)
    
    elif self.operation == "none":
      dec_x6 = dec_x6
      dec_y6 = dec_y6
      dec_z6 = dec_z6
      
      # pass the modified decoder outout to the Post convolution layer
      out_x = self.PostConv(dec_x6)
      out_y = self.PostConv(dec_y6)
      out_z = self.PostConv(dec_z6)

    # conlayer1 = self.conlayer1(dec_x5)
    # conlayer2 = self.conlayer2(dec_y5)
    # conlayer3 = self.conlayer3(dec_z5)

    out = torch.cat((out_x, out_y, out_z), 1)

    return out