from tkinter import Y
from matplotlib.pyplot import axis
from sympy import Float
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import math
import matplotlib.pylab as plt
from sklearn import preprocessing


class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train", force_scaler=None, I_list_scaler=None): #width_scaler=None, height_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split
        self.force_scaler = force_scaler
        #self.width_scaler = width_scaler
        #self.height_scaler = height_scaler
        self.I_list_scaler = I_list_scaler

    def __len__(self):
        return len(self.data)
    


    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

        force = []
        #width = []
        #height = []
        rigidity = []
        I_list = []

        with open(f'{full_path}/Rot_Stiff.txt','r') as f:
            for i, line in enumerate(f):
                    F, w, h, wc, hc, rig = line.split(',')
                    I = float(h)*1.0
                    #width.append(float(w))
                    #height.append(float(h))
                    I_list.append(float(I))
                    force.append(float(F))
                    rigidity.append(float(rig))

        if self.force_scaler != None:
            force = self.force_scaler.transform(np.array(force).reshape(-1,1))
            force = torch.from_numpy(force).float().squeeze(0)       
        else:
            force = torch.tensor(force)

        if self.I_list_scaler != None:
            I_list = self.I_list_scaler.transform(np.array(I_list).reshape(-1,1))
            I_list = torch.from_numpy(I_list).float().squeeze(0)       
        else:
            I_list = torch.tensor(I_list)


        # if self.width_scaler != None:
        #     width = self.width_scaler.transform(np.array(width).reshape(-1,1))
        #     width = torch.from_numpy(width).float().squeeze(0)       
        # else:
        #     width = torch.tensor(width)

        # if self.height_scaler != None:
        #     height = self.height_scaler.transform(np.array(height).reshape(-1,1))
        #     height = torch.from_numpy(height).float().squeeze(0)       
        # else:
        #     height = torch.tensor(height)
        
        rigidity = torch.tensor(rigidity)
        rigidity /= 100000


        #return force, width, height, rigidity
        return force, I_list, rigidity


def main():

    dataset = GridDataset(split='train')
    #force, width, height, rigidity = dataset[0]
    force, I_list, rigidity = dataset[0]

    print(force, I_list, rigidity)

    

if __name__ == "__main__":
    main()