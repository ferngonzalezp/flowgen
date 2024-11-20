import torch 
import torch.nn as nn
import torchvision
from model.block import*

class Bottleneck(nn.Module):
    
    """
    This class is to create bottleneck block to introduce inital parmaeters in the network
    
    Args:
        
        bottInputshape: shape of the concatenation of input parameters and encoder output as input of the bottleneck layer
        encoder_output_shape: output shape of the bottleneck layer
        y: input to bottleneck layer -> concatenatination of encoder output and inital paramters 
        
    Returns:
    """
    
    def __init__(self, bottInputshape, encoder_output_shape):
        super().__init__()
        
        self.dense = nn.Linear(bottInputshape, encoder_output_shape)

        self.activation1 = nn.ReLU()
        
        self.dropLayer = nn.Dropout()
        
        
        self.bn_after = nn.BatchNorm1d(bottInputshape)
        
        self.dense2 = nn.Linear(encoder_output_shape, encoder_output_shape)
        self.activation2 = nn.ReLU()


    def forward(self, y):
       
        # pass the encoder output to linear layer, activation and drouput layer
        y = self.dense(y.type(torch.float32))
        y = self.activation1(y)
        y = self.dropLayer(y)

        # Pass inear layer output through batch normalization
        # y = self.bn_after(y)        
        
        # pass the batch norm output through another linear layer and activation in the bottlenek block
        y = self.dense2(y)
        y = self.activation2(y)

        return y

        