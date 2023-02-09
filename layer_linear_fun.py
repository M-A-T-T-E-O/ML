# Modules 
import torch
import torch.nn as nn
########################################


# Implementation of y = wx + b

def apply_linear_layer(input_size, output_size, data):

 # Create new layer linear
 lay1 = nn.Linear(in_features=input_size, out_features=output_size)

 # Calculate the output from the layer linear
 y1 = lay1(data)

 # Verify for each node if the output calculation is correct (tollerance set to 0.001)
 for i in range(0,output_size):
  a = round((y1.data[i].item()),3)
  b = round((torch.matmul(lay1.weight.data[i,:], data) + lay1.bias[i]).data.item(),3)
  if a == b:
   if (i == (output_size-1)):
    res1 = "correct"
  else:
   if ( abs(a-b) > 0.001):
    res1 = "uncorrect"
    
 return y1, lay1, data, res1
