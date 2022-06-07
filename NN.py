from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, layer_sizes, dropout=0.1):
        super(NeuralNet, self).__init__()
        layers = []
        input_features = 2
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            #layers.append(nn.BatchNorm1d(output_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_features = output_features
        self.layers = nn.Sequential(*layers)


    #def forward(self, force, width, height):
    def forward(self, force, I_list):  # [B,1] [B,1] 
  
        
        B = force.shape[0]
        #a = torch.cat((force, width, height), dim = 1)
        a = torch.cat((force, I_list), dim = 1)       # [B,2]
        pred = self.layers(a)                                # [B,1]
       
   
        return pred
