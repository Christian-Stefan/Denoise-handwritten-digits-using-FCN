### ---- START M.L. Specific Imports --- ###
# from xml.parsers.expat import model
from assignment1_data.mnist_dataloader import create_dataloaders
import torch
import torch.nn as nn
import tensorflow as tf
from torch.optim import SGD
from torchvision import transforms
from torchvision.transforms import v2
### ---- END M.L. Specific Imports --- ###

### ---- START General Imports --- ###
import matplotlib.pyplot as PLT
import numpy as np
import os
import csv
import pandas as pd
from env_ignite import *
### ---- End General Imports --- ###


@staticmethod
def flattener(data:tuple):
    f = nn.Flatten(start_dim=0)
    flattened = [f(item) if isinstance(item, torch.Tensor) else item for item in data[:-1]]
    return tuple(flattened) + (data[-1],)
    
@staticmethod
def sanity_check(batch_size, loc):

    noisy_train, noisy_test, noisy_val = create_dataloaders(data_loc=loc, batch_size=batch_size)
    sampple_train, sample_test, sample_val = next(enumerate(noisy_train))[1][0], next(enumerate(noisy_test))[1][0], next(enumerate(noisy_val))[1][0]


    # Informative statement 1
    print("Three Data-Loaders were created\n" \
    "noisy_train of type {} with length {} batches, having extreme point of (min: {} max: {}) and both, variance = {} and mean = {}\n" \
    "noisy_test of type {} with length {} batches, having extreme point of (min: {} max: {}) and both, variance = {} and mean = {}\n" \
    "noisy_val of type {} with length {} batches, having extreme point of (min: {} max: {}) and both, variance = {} and mean = {}:".format(
                                                type(noisy_train), len(noisy_train), torch.min(sampple_train), torch.max(sampple_train), torch.mean(sampple_train), torch.var(sampple_train),
                                                type(noisy_test), len(noisy_test), torch.min(sample_test), torch.max(sample_test), torch.mean(sample_test), torch.var(sample_test),
                                                type(noisy_val), len(noisy_val), torch.min(sample_val), torch.max(sample_val), torch.mean(sampple_train), torch.var(sampple_train)

    ))

    no_samples: int = 5
    fig, axis = PLT.subplots(no_samples, 6)   # 3 splits x (clean,noisy) = 6 columns

    ex_tr, ex_te, ex_va = enumerate(noisy_train), enumerate(noisy_test), enumerate(noisy_val)
    Ex: list = [ex_tr, ex_te, ex_va]

    for sample in range(no_samples):
        for index, ex in enumerate(Ex):
            _, (clean_sample, noisy_sample, label) = next(ex)

            col = 2 * index  # <- each split gets two dedicated columns

            # fig.suptitle("")
            axis[sample, col].imshow(clean_sample[0].reshape(32, 32), cmap="gray")
            axis[sample, col].set_title(f"Label {label[0]}")
            axis[sample, col].axis("off")

            axis[sample, col + 1].imshow(noisy_sample[0].reshape(32, 32), cmap="gray")
            axis[sample, col + 1].set_title("Noisy Sample")
            axis[sample, col + 1].axis("off")

    PLT.tight_layout()
    PLT.show()

class BasicFCN(nn.Module):
    def __init__(self, width: list, activation: list):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # 1. Identify activation classes from strings
        hid_actv_class = getattr(nn, activation[0])
        final_actv_class = getattr(nn, activation[1])
        
        # 2. Iterate through the width list to create hidden layers
        # ... width[:-2] handles everything up to the penultimate layer
        for i in range(len(width) - 2):
            self.layers.append(nn.Linear(width[i], width[i+1]))
            self.layers.append(nn.BatchNorm1d(width[i+1]))
            
            # 2.1 Handle LeakyReLU's negative_slope if necessary
            if activation[0] == "LeakyReLU":
                self.layers.append(hid_actv_class(negative_slope=0.01))
            else:
                self.layers.append(hid_actv_class())

        # 3. The Output Head (Final Layer)
        self.output_layer = nn.Linear(width[-2], width[-1])
        self.final_activation = final_actv_class()

    def forward(self, x):
        # 4. Flatten image if it comes in as (Batch, 32, 32)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # 5. Run through all generated hidden layers
        for layer in self.layers:
            x = layer(x)
            
        # 6. Run through output head
        x = self.output_layer(x)
        x = self.final_activation(x)
        return x

class Noisy_Image_Identifier_Net(nn.Module):

    """
    The current models holds ReLu as non-linear activation units.
    The network should be a class and its constructor `__init__` should accept an argument defining the width of the network (e.g., layer sizes) - Done

    e.g., YourClassName([32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]) - Done
    - 32^2 inputs (one image at a time); - Done
    - multiple hidden layers (24**2, 22**2, 20**2, 22**2, 24**2, 32**2) - Done
    - 32^2 outputs(the reconstruction of the image) - Done
    - 1 output (the classification head)
    """
    def __init__(self, width:list, activation:list):
        super().__init__() # Calling in the constructor of the inhereted class `nn.Module`
        self.width = width
        hid_actv_function = getattr(nn, activation[0])
        final_actv_function = getattr(nn, activation[1])

        # ...start building the network within the constructor 
        self.fcn_input = nn.Linear(in_features=width[0], out_features=width[1], bias=True)
        self.btcnorm1 = nn.BatchNorm1d(width[1])
        self.activation1 = hid_actv_function(0.01)

        self.fcn_hidlay1 = nn.Linear(in_features=width[1], out_features=width[2], bias=True)
        self.btcnorm2 = nn.BatchNorm1d(width[2])
        self.activation2 = hid_actv_function(0.01)

        self.fcn_hidlay2 = nn.Linear(in_features=width[2], out_features=width[3], bias=True)
        self.btcnorm3 = nn.BatchNorm1d(width[3])
        self.activation3 = hid_actv_function(0.01)

        self.fcn_hidlay3 = nn.Linear(in_features=width[3], out_features=width[4], bias=True)
        self.btcnorm4 = nn.BatchNorm1d(width[4])
        self.activation4 = hid_actv_function(0.01)
       
        self.fcn_hidlay4 = nn.Linear(in_features=width[4], out_features=width[5], bias=True)
        self.btcnorm5 = nn.BatchNorm1d(width[5])
        self.activation5 = hid_actv_function(0.01) # TODO Argue the reason that made you addopt `0.01`

        self.fcn_hidlay5 = nn.Linear(in_features=width[5], out_features=width[6], bias=True)
        self.btcnorm6 = nn.BatchNorm1d(width[6])
        self.activation6 = hid_actv_function(0.01) # TODO Argue the reason that made you addopt `0.01`

        self.fcn_hidlay6 = nn.Linear(in_features=width[6], out_features=width[7], bias=True)
        self.activation7 = final_actv_function()
  
    def forward(self, x):
        x = self.activation1(self.btcnorm1(self.fcn_input(x)))
        x = self.activation2(self.btcnorm2(self.fcn_hidlay1(x)))
        x = self.activation3(self.btcnorm3(self.fcn_hidlay2(x)))
        x = self.activation4(self.btcnorm4(self.fcn_hidlay3(x)))
        x = self.activation5(self.btcnorm5(self.fcn_hidlay4(x)))
        x = self.activation6(self.btcnorm6(self.fcn_hidlay5(x)))
        
        # x = self.fcn_hidlay5(x) 
        # x = self.activation6(x) 
        
        # The Output "Head"
        x = self.fcn_hidlay6(x) 
        x = self.activation7(x)
        
        # Debug Statement 4
        # print("Last layer shape {}".format(x.shape))
        return x


def training(model, loc, batch_size, basisc_transf, activation, width, epochs, model_label, saving_path, AUTOMATIC=True):

    if AUTOMATIC:
        data = pd.read_csv(saving_path)
        PLT.plot(data['train_loss'], label = 'Train loss')
        PLT.plot(data['test_loss'], label='Test Loss')
        PLT.xlabel("Epochs")
        PLT.ylabel("Loss expressed in MSE")
        PLT.title(f"Train and Validation losses of {model_label} FCN")
        PLT.grid()
        PLT.legend()
        PLT.show()
        return True
        # Plot the results from this location...

    # 1. Loading data
    train, test, val = create_dataloaders(data_loc=loc, batch_size=batch_size, transform=basisc_transf)

    model = BasicFCN(activation=activation, width=width)
    optimizer = SGD(params=model.parameters(), lr=0.01) # optimizer defined outside of the training loop
    loss = nn.MSELoss()
    model_label:str = model_label
    if model_label != None and ('weights_denoiser' in os.listdir()) == False:
        os.makedirs('weights_denoiser')
        if (model_label in os.listdir('weights_denoiser')) == False:
            os.makedirs(f'weights_denoiser\{model_label}')
            os.makedirs(f'weights_denoiser\{model_label}_accuracy_container')

    # 3. Training & Validation kick-off
    # 3.1 Creating history-holder containers
    loss_container_train, loss_container_te = [], []
    for epoch in range(epochs):
        _loss_per_epoch_tr:float = 0.0
        _loss_per_epoch_te:float = 0.0
        #TODO Optimize the looping mechanism for faster iteration;

        # 3.2 Training kicks off
        model.train()
        for cl, ns, label in train:
            optimizer.zero_grad()
            output = model(ns)
            # 3.3 Quantifying the quality of the output; Backpropagate; Optimizer step
            _loss_tr = loss(output,cl) # Compare the batch of recon against the batch with clear
            _loss_tr.backward()
            optimizer.step()
            # 3.4 Keep track of performance
            _loss_per_epoch_tr+=_loss_tr.item()

        loss_container_train.append(_loss_per_epoch_tr/len(train))
        # 4. Testing kicks off
        model.eval()
        with torch.no_grad():
            for cl, ns, label in test:
                output = model(ns)
                _loss_te = loss(output, cl)
                _loss_per_epoch_te+=_loss_te.item()
            loss_container_te.append(_loss_per_epoch_te/len(test))

        print("Epoch {} yielded: Train Loss {:.2f} | Test Loss {:.2f} ".format(epoch, loss_container_train[epoch], loss_container_te[epoch]))
        # 5. Ensures the weights are saved one epoch at a time;
        torch.save(model.state_dict(), f'weights_denoiser\{model_label}\{model_label}model_weights_epoch{epoch}.pth')
    # 6. Ensures both containers are being saved under `high-capacity_accuracy_container\history.csv`
    # 6.1 Convert the lists to a pandas dataframe for easier saving
    raw_history_frame = pd.DataFrame({
        "train_loss": loss_container_train, 
        "test_loss": loss_container_te
    })
    # 6.2 Save the dataframe as a .csv file
    raw_history_frame.to_csv(f"weights_denoiser\{model_label}_accuracy_container\history.csv")


    PLT.plot(loss_container_train, label = 'Train loss')
    PLT.plot(loss_container_te, label='Test Loss')
    PLT.xlabel("Epochs")
    PLT.ylabel("Loss expressed in MSE")
    PLT.title(f"Train and Validation losses of {model_label} FCN")
    PLT.grid()
    PLT.legend()
    PLT.show()


if __name__ == "__main__":

    # Import env variables
    from env_ignite import (
    defaultFCN_hist_path,
    defaultFCN_weight_path
)
    # Training variables
    batch_size:int = 64
    loc:str = r"/DataSets"
    epochs:int = 30

    # Mini pipeline
    basisc_transf = v2.Compose([
    # transforms.ToTensor(), <- Data is of tensor nature by definition
    # v2.Normalize(mean=[0.5],std=[0.5]),
    flattener
])


    # Default Model ------------------------------ #
    width_default:list = [32**2, 24**2, 16**2, 32**2]
    hidd_activation_default:str = "ReLU"
    final_activation_default:str = "Sigmoid"
    activation_def:list = [hidd_activation_default, final_activation_default]


    # Complex Model ------------------------------ #
    width_complex:list = [32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2]
    hidd_activation_comp:str = "LeakyReLU"
    final_activation_comp:str = "Tanh"
    activation_comp:list = [hidd_activation_comp, final_activation_comp]

    # Low Model ------------------------------ #
    width_low:list = [32**2, 16**2, 32**2]
    hidd_activation_low:str = "ReLU"
    final_activation_low:str = "Sigmoid"
    activation_low:list = [hidd_activation_low, final_activation_comp]

    
    # 1.1.a)
    batch_size:int = 64
    sanity_check(batch_size=batch_size, loc=loc)

    # 1.1.g)
    model = BasicFCN(activation=activation_def, width=width_default)
    # training(model, loc, batch_size, basisc_transf, 
    #                     activation_def, width_default, epochs, 
    #                     model_label="Default Model", AUTOMATIC=False, saving_path=defaultFCN_hist_path)
    
    training(model, loc, batch_size, basisc_transf, 
                        activation_comp, width_complex, epochs, 
                        model_label="Complex Model", AUTOMATIC=False, saving_path=defaultFCN_hist_path)
   

    training(model, loc, batch_size, basisc_transf, 
                            activation_low, width_low, epochs, 
                            model_label="Low Model", AUTOMATIC=False, saving_path=defaultFCN_hist_path)
   
   
  

    
    



    

    
