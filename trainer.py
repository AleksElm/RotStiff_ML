from matplotlib import projections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import tqdm
import wandb
from NN import NeuralNet
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from sklearn import preprocessing
import math

def _RELMSE(img, target, eps=0.1):
    nom = (img - target) * (img - target)
    denom = img * img + target * target + 0.1 
    return torch.mean(nom / denom)

@torch.no_grad()
def no_grad_loop(data_loader, model, iter, epoch=2, device="cuda", batch_size = 64):

    no_grad_loss = 0
    l1_loss = 0
    cnt = 0

    for i, (force, I_list, rigidity) in enumerate(data_loader):

        # transfer data to device
        force = force.to(device)
        #width = width.to(device)
        #height = height.to(device)
        I_list = I_list.to(device)
        rigidity = rigidity.to(device)

        with autocast():
            
            rigidity_pred = model(force, I_list)
            #loss = _RELMSE(rigidity_pred, rigidity)
            loss = F.l1_loss(rigidity_pred, rigidity)

        no_grad_loss += loss
        #l1_loss += loss_l1
        cnt += 1

        if i == len(data_loader) -1:
        
            case_0 = 0
            rigidity_pred_0 = rigidity_pred[case_0]
            rigidity_0 = rigidity[case_0]

            case_1 = 1
            rigidity_pred_1 = rigidity_pred[case_1]
            rigidity_1 = rigidity[case_1]

            case_2 = 2
            rigidity_pred_2 = rigidity_pred[case_2]
            rigidity_2 = rigidity[case_2]             

            wandb.log({'rigidity(0)': rigidity_0, 'rigidity_pred(0)': rigidity_pred_0})
            wandb.log({'rigidity(1)': rigidity_1, 'rigidity_pred(1)': rigidity_pred_1})
            wandb.log({'rigidity(2)': rigidity_2, 'rigidity_pred(2)': rigidity_pred_2})

    return no_grad_loss/cnt#, l1_loss/cnt


def train(model: NeuralNet, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=1e-3, device="cuda"):
    
    iter = 0
    valid_loss = no_grad_loop(validation_loader, model, iter, epoch=0, device="cuda", batch_size=batch_size)
    curr_lr =  learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.1, patience=2)

# Early Stopping
    min_loss = 100
    patience = 6
    trigger_times = 0

    # training loop
    training_losses = {
        "train": {},
        "valid": {}
    }
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        loader = tqdm.tqdm(train_loader)
        for force, I_list, rigidity in loader:
            # transfer data to device
            force = force.to(device)
            I_list = I_list.to(device)
            #width = width.to(device)
            #height = height.to(device)
            rigidity = rigidity.to(device)

            #print(force, I_list, rigidity)
            

            # Forward pass
            with autocast():
                
                rigidity_pred = model(force, I_list)
                loss = F.l1_loss(rigidity_pred, rigidity)
               

            loader.set_postfix(loss = loss.item())

            # Backward and optimize
            optimizer.zero_grad()               # clear gradients
            scaler.scale(loss).backward()       # calculate gradients

            # grad less than 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            scaler.step(optimizer)
            scaler.update()
            iter += 1

            training_losses["train"][iter] = loss.item()

            if (iter+1) % 20 == 0:

                # validation loop
                model = model.eval()
                valid_loss = no_grad_loop(validation_loader, model, iter, epoch, device="cuda", batch_size=batch_size)
         
                curr_lr =  optimizer.param_groups[0]["lr"]
                wandb.log({"valid loss": valid_loss.item(), "lr": curr_lr}, commit=False)
                model = model.train()
            wandb.log({"train loss": loss.item()})


            if (iter+1) % 200 == 0:

                # validation loop
                model = model.eval()
                valid_loss = no_grad_loop(validation_loader, model, iter, epoch, device="cuda", batch_size=batch_size)

                if valid_loss > min_loss:
                    trigger_times += 1
                    min_loss = valid_loss

                    if trigger_times >= patience:
                        return model

                else: 
                    trigger_times = 0

                
                scheduler.step(valid_loss)

         
                #curr_lr =  optimizer.param_groups[0]["lr"]
                #wandb.log({"valid loss": valid_loss.item(), "lr": curr_lr}, commit=False)
                #model = model.train()
            #wandb.log({"train loss": loss.item()})


    # test loop
    test_loss = no_grad_loop(test_loader, model, iter, device="cuda", batch_size=batch_size)
    print(f'testloss: tot={test_loss:.5f}')
    return model