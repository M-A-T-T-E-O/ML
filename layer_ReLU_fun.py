# Modules 
import torch
import torch.nn as nn
########################################


# Implementation of non-linear layer ReLU

def apply_ReLU(y1, output_size):

 # Create non-linear layer ReLU
 lay2 = nn.ReLU()

 # Calculate the output from the layer ReLU
 y2 = lay2(y1) 

 # Verify for each node if the output calculation is correct
 for i in range(0,output_size):
  if ((y1[i].data.item() < 0 and y2[i].data.item() == 0) or (y1[i].data.item() == y2[i].data.item())):
   if (i == (output_size - 1)):
    res2 = "correct"
  else:
   res2 = "uncorrect"    

 return y2, lay2, res2


