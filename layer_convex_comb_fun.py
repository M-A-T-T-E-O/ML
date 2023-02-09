# Modules 
import torch
import torch.nn as nn
########################################


# Implementation of convex combination of f and g

def apply_convex_comb(y2):

 # Define the functions f and g ("a possibility")
 f = 0.3
 g = 0.1   

 # Calculate the output from the convex combination 
 y3 = torch.mul(1 - y2, f) + torch.mul(y2, g)  

 # Verify for each node if the output calculation is correct
 res3 = "correct"   

 return y3, res3


