import torch 
import torch.nn as nn
import torchvision
from model.block import*

class Encoder(nn.Module):
  """
  Class to create Encoder in the network
  Args:
      in_ch: number of input channels
      out_ch: number of output channels
      kernel_size: kernel size
      x: input to encoder
  Returns:
  """
  def __init__(self, in_ch, out_ch, kernel_size,same_size):
    super().__init__()
    self.enc_blocks = Block(in_ch, out_ch, kernel_size, same_size)

  def forward(self, x):
    return self.enc_blocks(x)

