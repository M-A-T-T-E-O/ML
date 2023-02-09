# Modules 
import torch
import torch.nn as nn
import numpy as np
########################################

# Function imported
from layer_linear_fun import apply_linear_layer
from layer_ReLU_fun import apply_ReLU
from layer_convex_comb_fun import apply_convex_comb
from trainingNN_fun import trainingNN
from layer_class import Layer


# Implementation for the forward Neural network

# Initialize variables
layer_list = []
str_layers = ""
torch.manual_seed(1234) # Initialize the pseudorandom generator with a seed

# Initialize sentinels 
ans = "y" # yes
first = 1
num_layers = 0

while (ans == "y" or first == 1):      
     
 # Get from standard input the output/input size
 input_size = int(input("Please insert the size of input (Layer " + str(num_layers) + ") :"))
 output_size = int(input("Please insert the size of output (Layer " + str(num_layers) + ") :"))
 
 # Generate the sequence (the dimension is equal to input_size) of random numbers in [0,1) 
 if (first == 1):
  data = torch.rand(input_size) 

 # Add the new layer just created into array   
 if (first == 1):
  [y1, lay1, x, res1] = apply_linear_layer(input_size, output_size, data)
  [y2, lay2, res2] = apply_ReLU(y1, output_size)
  [y3, res3] = apply_convex_comb(y2)  
 else:
  [y1, lay1, x, res1] = apply_linear_layer(input_size, output_size, layer_list[num_layers-1].y3)    
  [y2, lay2, res2] = apply_ReLU(y1, output_size)
  [y3, res3] = apply_convex_comb(y2)  
 lay = Layer(y1, y2, y3, lay1, lay2, x, res1, res2, res3)
 layer_list.append(lay)
 str_layers = str_layers + "Layer " + str(num_layers) + " -> " 
 num_layers = num_layers + 1  
 first = 0  
 
 # The model
 print()
 print("A priori conditions:")
 print("------------------------------")
 print("The output from linear model is", lay.res1)
 print("------------------------------")
 print("The output from ReLU is", lay.res2)
 print("------------------------------")
 print("The output from convex combination is", lay.res3)   
 print()
 
 print("Model parameters:")
 print("------------------------------")
 print("The x (data): ",lay.x.data)
 print("------------------------------")
 print("The w (weight): ",lay.lay1.weight.data)
 print("------------------------------")
 print("The b (bias): ",lay.lay1.bias.data)
 print()
 
 print("First level output:")
 print("------------------------------")
 print("The y_1 (output from lin. model):",lay.y1.data)
 print()
 
 print("Second level output:")
 print("------------------------------")
 print("The y_2 (output from ReLU):",lay.y2.data)
 print()
    
 print("Third level output:")
 print("------------------------------")
 print("The y_3 (output from convex combination):",lay.y3.data)
 print()   
 print("----------------------------------------------------------------------------------------------------------------------")
 print()
 print("My Neural Network is: input ->", str_layers ,"output" )
 print()
 ans = str(input("Do you want to add a new Layer? (type ""y"" (yes) or ""n"" (no)?)"))
 while (ans != "n" and ans != "y"):
  ans = str(input("Do you want to add a new Layer? (type ""y"" (yes) or ""n"" (no)?)"))  
  




