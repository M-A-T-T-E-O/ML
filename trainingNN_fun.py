# Modules 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
########################################

# Function imported
from torch import optim
from layer_convex_comb_fun import apply_convex_comb


# Training algorithm for the Neural Network

def trainingNN(layer_list_NN):
 
 # The input data can be choosen randomly ("a possibility")
 input_data = torch.randint(0,10,(100,1), dtype=torch.float)

 # The output data are supposed as follows ("a possibility")
 output_data = torch.empty(input_data.shape)
 #output_data = torch.empty(100,2)
 output_data[:,0] = torch.exp(input_data[:,0])
 #output_data[:,1] = torch.exp(input_data[:,0]*0)
 #output_data[:,1] = input_data[:,0] + input_data[:,1]            
 #output_data[:,2] = input_data[:,0] + input_data[:,1] + input_data[:,2] 
    

 # Define the Mean Squared Error 
 loss = torch.nn.MSELoss()

 # Define the optimizer
 param_list_nn = list()  
 for n in range (0,len(layer_list_NN)):
  param_list_nn = param_list_nn + list(layer_list_NN[n].lay1.parameters())
 optimizer = optim.SGD(param_list_nn,lr=1e-5) 
    
 # Decide how to define each output y_i
 print ("Each output is calculated in maximum 3 steps as follows:\n\ny0 = Ax + B\n\ny1 = ReLU(y0)\n\ny2 = (1 - y1)*f(x) + y1*g(x)")
 out_def = ""   
 print()
 while (out_def != "0" and out_def != "1" and out_def != "2"):
  out_def = str(input("What output do you want to take for the tuning of parameters (type 0 (y0) or 1 (y1) or 2 (y2))?\n"))
 print()
       
 # Tuning of the matrices (both weight A and bias B)). 
 # The model is Ax + B = y, where: 
 # x = (input_data)' ( 1000x3 matrix ) 
 # y = (y0 y1 ... y999) ( 3x1000 matrix ), yn = output n associated to the input n (3x1 vector)  
 # B = (b b ... b) ( 3x1000 matrix)   
 for i in range(300000):   
 # Calculate the output of the Neural Network from the given input dataset 
  for k in range(0,len(layer_list_NN)):
   if (k == 0): 
    if (out_def == "0"):
     ynn = layer_list_NN[0].lay1(input_data)                                               # linear function
    elif (out_def == "1"):
     ynn = layer_list_NN[0].lay2(layer_list_NN[0].lay1(input_data))                        # linear function + ReLU
    else:
     ynn = apply_convex_comb(layer_list_NN[0].lay2(layer_list_NN[0].lay1(input_data)))     # linear function + ReLU + convex combination     
   else:
    if (out_def == "0"):
     ynn = layer_list_NN[k].lay1(ynn)
    elif (out_def == "1"):
     ynn = layer_list_NN[k].lay2(layer_list_NN[k].lay1(ynn))
    else:  
     ynn = apply_convex_comb(layer_list_NN[k].lay2(layer_list_NN[k].lay1(ynn[0])))   
  if (out_def == "2"):
   ynn = ynn[0]
  
  # Calculate the error between the target and the output
  error = loss(ynn, output_data)    
    
  # Calculate the gradient for each tensor of weight and bias
  error.backward()  
  
  # Update the parameters
  optimizer.step() 
  optimizer.zero_grad()
    
  # Print the error every 250 iterations
  if (i == 0):
   print("The error (every 250 steps):")     
  if np.mod(i,250)==0:
   print(error)
 
 # Plot graphs to compare target output with the NN output
 # sample = np.repeat(np.arange(0, 100), 1)
 # plt.plot(sample, output_data, color ='tab:blue',label="target output") 
 # plt.plot(sample, ynn.data, color ='tab:orange',label="NN output")
 # plt.title('Target output Vs NN output') 
 # plt.legend(loc="upper left")


 return print("\nThe output of the NN (post tuning) is:\n",ynn.data,"\n","The target output (from measurements) is:\n",output_data) 


