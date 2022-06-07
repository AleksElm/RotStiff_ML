from tabnanny import verbose
from sympy import re
import wandb
from NN import NeuralNet
import torch
from dataset import GridDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd, dropout
import numpy as np
import random 
import wandb
from sklearn import preprocessing


# Fixed seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getAllData():
    training_dataset = GridDataset()
    data_force = torch.stack([d[0] for d in training_dataset]).numpy()
    data_I_list = torch.stack([d[1] for d in training_dataset]).numpy()
    #data_width = torch.stack([d[1] for d in training_dataset]).numpy()
    #data_height = torch.stack([d[1] for d in training_dataset]).numpy()

    #return data_force, data_width, data_height
    return data_force, data_I_list

def run():
    layer_sizes = [256,256,1]
    num_epochs = 200
    batch_size = 64
    batch_size_train = 64
    learning_rate = 0.005
    dropout = 0.1

    dict = {
        'learning_rate': learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "layer_sizes" : layer_sizes,
    
    }

    use_existing_model = False

    #data_force, data_width, data_height = getAllData()
    data_force, data_I_list = getAllData()

    forces_scaler = preprocessing.MinMaxScaler().fit(data_force)
    I_list_scaler = preprocessing.MinMaxScaler().fit(data_I_list)
    #width_scaler = preprocessing.StandardScaler().fit(data_width)
    #height_scaler = preprocessing.StandardScaler().fit(data_height)
    
    # Dataset
    train_dataset = GridDataset(force_scaler=forces_scaler, I_list_scaler=I_list_scaler)  # width_scaler=width_scaler, height_scaler=height_scaler)
    test_dataset = GridDataset(split="test",force_scaler=forces_scaler, I_list_scaler=I_list_scaler)  # width_scaler=width_scaler, height_scaler=height_scaler)
    validation_dataset = GridDataset(split="validation",force_scaler=forces_scaler, I_list_scaler=I_list_scaler)  # width_scaler=width_scaler,height_scaler=height_scaler)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,  drop_last=True, num_workers=3, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)



    model = NeuralNet(layer_sizes, dropout=dropout).to(device)
    if use_existing_model:
        model.load_state_dict(torch.load("./003model.pth")["state_dict"])
    wandb.init(project="rigidity_pred", entity="master-thesis-ntnu", config=dict)
    wandb.watch(model, log='all', log_freq=10)
    model = train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)


    # Save model
    config = {
        "state_dict": model.state_dict()
    }

    torch.save(config, "model.pth")

if __name__ == "__main__":
    run()