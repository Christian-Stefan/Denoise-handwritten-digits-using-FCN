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
import ctypes
### ---- End General Imports --- ###

@staticmethod
def flattener(data:tuple):
    f = nn.Flatten(start_dim=0)
    flattened = [f(item) if isinstance(item, torch.Tensor) else item for item in data[:-1]]
    return tuple(flattened) + (data[-1],)

import torch
import torch.nn as nn

class Noisy_Image_Identifier_Net(nn.Module):
    def __init__(self, width: list, activation: list):
        """
        Args:
            width (list): List of layer sizes, e.g., [1024, 576, 484, 400, 484, 576, 1024, 1]
            activation (list): Strings of activation names, e.g., ["LeakyReLU", "Sigmoid"]
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # 1. Identify activation classes from strings
        hid_actv_class = getattr(nn, activation[0])
        final_actv_class = getattr(nn, activation[1])
        
        # 2. Iterate through the width list to create hidden layers
        # This loop handles all layers except the very last one (the head)
        for i in range(len(width) - 2):
            # Add the Linear transformation
            self.layers.append(nn.Linear(width[i], width[i+1], bias=True))
            
            # Add Batch Normalization
            self.layers.append(nn.BatchNorm1d(width[i+1]))
            
            # Add the Hidden Activation
            # We use 0.01 specifically for LeakyReLU to prevent "dying neurons" 
            # by allowing a small gradient for negative inputs.
            if activation[0] == "LeakyReLU":
                self.layers.append(hid_actv_class(negative_slope=0.01))
            else:
                self.layers.append(hid_actv_class())

        # 3. The Output Head (Classification/Identification Layer)
        # Maps the penultimate width to the final output (e.g., 1 for classification)
        self.output_layer = nn.Linear(width[-2], width[-1])
        self.final_activation = final_actv_class()

    def forward(self, x):
        # 4. Flatten input if it's a 2D image (Batch, Height, Width) -> (Batch, Flat)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # 5. Iterative forward pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            
        # 6. Final output head
        x = self.output_layer(x)
        x = self.final_activation(x)
        
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
        train_loader, test_loader, val_loader = create_dataloaders(data_loc=loc, batch_size=batch_size, transform=basisc_transf)

        model = Noisy_Image_Identifier_Net(activation=activation, width=width)
        optimizer = SGD(params=model.parameters(), lr=0.005) # optimizer defined outside of the training loop
        loss = nn.BCELoss()

        model_label:str = model_label
        if model_label != None and ('weights_denoiser' in os.listdir()) == False:
            os.makedirs('weights_denoiser')
            if (model_label in os.listdir('weights_denoiser')) == False:
                os.makedirs(f'weights_denoiser\{model_label}')
                os.makedirs(f'weights_denoiser\{model_label}_accuracy_container')

        # 3. Training & Validation kick-off
        # 3.1 Creating history-holder containers
        loss_container_train, loss_container_te, correctly_predicted_images  = [], [], []
        for epoch in range(epochs):
            _loss_per_epoch_tr:float = 0.0
            _loss_per_epoch_te:float = 0.0
            correct_images:int = 0

            #TODO Optimize the looping mechanism for faster iteration;
            # 3.2 Training kicks off
            model.train()
            for cl, ns, label in train_loader:
                # 4. Craft new data loaders
                # 4.1 Extract the batch size dynamically to create the labels tensor
                current_b_size = cl.shape[0]
                dummy_torch = torch.empty(current_b_size,1)
                # 4.2 Concatenate the clean and noisy images to create a new batch of double size
                conct_img_train = torch.cat((cl, ns))
                conct_labels_train = torch.cat((torch.ones_like((dummy_torch)), torch.zeros_like((dummy_torch))))

                # 4.3 Shuffle the concatenated batch to avoid overfitting;
                shuffle_indices = torch.randperm(conct_img_train.size(0))
                conct_img_train = conct_img_train[shuffle_indices]
                conct_labels_train = conct_labels_train[shuffle_indices]

                # 5. Forward pass and loss computation
                optimizer.zero_grad()
                output = model(conct_img_train)
                # 6. Quantifying the quality of the output; Backpropagate; Optimizer step
                _loss_tr = loss(output, conct_labels_train) # Compare the batch of recon against the batch with clear
                _loss_tr.backward()
                optimizer.step()
                # 7. Keep track of performance
                _loss_per_epoch_tr+=_loss_tr.item()

            loss_container_train.append(_loss_per_epoch_tr/len(train_loader))
            # 4. Testing kicks off
            model.eval()
            with torch.no_grad():
                for cl, ns, label in test_loader:
                    # 4. Craft new data loaders
                    # 4.1 Extract the batch size dynamically to create the labels tensor
                    current_b_size = cl.shape[0]
                    dummy_torch = torch.empty(current_b_size,1)
                    
                    # 4.2 Concatenate the clean and noisy images to create a new batch of double size
                    conct_img_test = torch.cat((cl, ns))
                    conct_labels_test = torch.cat((torch.ones_like((dummy_torch)), torch.zeros_like((dummy_torch))))
                    
                    # 4.3 Shuffle the concatenated batch to avoid overfitting;
                    shuffle_indices = torch.randperm(conct_img_test.size(0))
                    conct_img_test = conct_img_test[shuffle_indices]
                    conct_labels_test = conct_labels_test[shuffle_indices]

                    # 6. Quantifying the quality of the output; Backpropagate; Optimizer step
                    output = model(conct_img_test)
                    _loss_te = loss(output, conct_labels_test)
                    _loss_per_epoch_te+=_loss_te.item()

                    # 7. Empirical (e.g., only for observation purposes) validation metric
                    # 7.1 Put in place the empirical validation criteiron (e.g., Counting the amount of correct instances the network classified)
                    criterion = torch.where(output>0.5, 1.0, 0.0)
                    # 7.2 Count the number of correctly predicted images on batch
                    correct_images += (criterion==conct_labels_test).sum()     
                    accuracy = (correct_images.item() / (len(test_loader.dataset) * 2)) * 100 
                loss_container_te.append(_loss_per_epoch_te/len(test_loader))
                correctly_predicted_images.append(accuracy)

            print(correct_images, len(test_loader.dataset)*2)
            print("Epoch {} yielded: Train Loss {:.5f} | Test Loss {:.5f} | Correctly predicted images in %{}".format(
                epoch, loss_container_train[epoch], loss_container_te[epoch], accuracy))
            
            # 8. Ensures the weights are saved one epoch at a time;
            torch.save(model.state_dict(), f'weights_denoiser\{model_label}\{model_label}model_weights_epoch{epoch}.pth')
        # 9. Ensures both containers are being saved under `high-capacity_accuracy_container\history.csv`
        # 9.1 Convert the lists to a pandas dataframe for easier saving
        raw_history_frame = pd.DataFrame({
            "train_loss": loss_container_train, 
            "test_loss": loss_container_te
        })
        # 6.2 Save the dataframe as a .csv file
        raw_history_frame.to_csv(f"weights_denoiser\{model_label}_accuracy_container\history.csv")

        # PLT.plot(loss_container_train, label = 'Train loss')
        # PLT.plot(loss_container_te, label='Test Loss')
        # PLT.plot(correctly_predicted_images, label='Correctly predicted images')
        # PLT.xlabel("Epochs")
        # PLT.ylabel("Loss expressed in MSE")
        # PLT.title(f"Train and Validation losses of {model_label} FCN")
        # PLT.grid()
        # PLT.legend()
        # PLT.show()


if __name__ == "__main__":
    import argparse
    from env_ignite import (
        defaultFCN_det_weight,
        defaultFCN_det_hist,
        compFCN_det_weight,
        compFCN_det_hist
    )

    # Training variables
    batch_size:int = 64
    loc:str = r"/DataSets"
    epochs:int = 30

    # Mini pipeline
    basisc_transf = v2.Compose([
    # transforms.ToTensor(), <- Data is of tensor nature by definition
    # v2.Normalize(mean=[0.5],std=[0.5]),
    flattener])

    # Default Model ------------------------------ #
    width_default:list = [32**2,  24**2, 16**2, 32**2, 1]
    hidd_activation_default:str = "ReLU"
    final_activation_default:str = "Sigmoid"
    activation_def:list = [hidd_activation_default, final_activation_default]


    # Complex Model ------------------------------ #
    width_complex:list = [32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]
    hidd_activation_comp:str = "LeakyReLU"
    final_activation_comp:str = "Sigmoid"
    activation_comp:list = [hidd_activation_comp, final_activation_comp]

    # Low Model ------------------------------ #
    width_low:list = [32**2, 1**2, 1**2, 1**2, 1**2, 1**2, 1**2, 1]
    hidd_activation_low:str = "ReLU"
    final_activation_low:str = "Sigmoid"
    activation_low:list = [hidd_activation_low, final_activation_comp]

    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load results/weights without retraining.")
    args = parser.parse_args()

    if args.load:
        print("Running in --load mode: Fetching Detector Results...")

        ctypes.windll.user32.MessageBoxW(0, "1+2). Two things are to be replaced in the previous architecture: Add a new output layer holding only one neuron and equipping it with a sigmoid function as activation output.","Answers to the first batch of questions",1)

        # 1.2.c)
        model = Noisy_Image_Identifier_Net(activation=activation_def, width=width_default)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Default Model Detector", AUTOMATIC=True, saving_path=defaultFCN_det_hist)

        model = Noisy_Image_Identifier_Net(activation=activation_comp, width=width_complex)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Complex Model Detector", AUTOMATIC=True, saving_path=compFCN_det_hist)

        model = Noisy_Image_Identifier_Net(activation=activation_low, width=width_low)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Low Model Detector", AUTOMATIC=True, saving_path=defaultFCN_hist_path)
        
        
        ctypes.windll.user32.MessageBoxW(0, "3+4). Having acquired a significantly high accuracy score after the completion of the first epoch prompted me to question and closely examine the training loop to see if I accidentally built in any flaws that might have caused an artificial inflation of the score and over-optimistic performance. However, after changing the noise argument in MNIST_loader to 0.1 to make the images hold less noise and thus make the detector's mission more difficult, I observed a major decrease in accuracy. Based on this, I concluded that irrespective of the model architecture differences between corrupted and not corrupted with noise images, it remains a trivial task when the injected noise comes in big quantities (e.g., noise = 0.5).","Answers to the second batch of questions",1)

    else:
        ctypes.windll.user32.MessageBoxW(0, "1+2). Two things are to be replaced in the previous architecture: Add a new output layer holding only one neuron and equipping it with a sigmoid function as activation output.","Answers to the first batch of questions",1)

        print("Running manually...")
        model = Noisy_Image_Identifier_Net(activation=activation_def, width=width_default)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Default Model Detector", AUTOMATIC=True, saving_path=defaultFCN_det_hist)

        model = Noisy_Image_Identifier_Net(activation=activation_comp, width=width_complex)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Complex Model Detector", AUTOMATIC=True, saving_path=compFCN_det_hist)

        model = Noisy_Image_Identifier_Net(activation=activation_low, width=width_low)
        training(model, loc, batch_size, basisc_transf, 
                            activation_def, width_default, epochs, 
                            model_label="Low Model Detector", AUTOMATIC=False, saving_path=defaultFCN_hist_path)

        ctypes.windll.user32.MessageBoxW(0, "3+4). Having acquired a significantly high accuracy score after the completion of the first epoch prompted me to question and closely examine the training loop to see if I accidentally built in any flaws that might have caused an artificial inflation of the score and over-optimistic performance. However, after changing the noise argument in MNIST_loader to 0.1 to make the images hold less noise and thus make the detector's mission more difficult, I observed a major decrease in accuracy. Based on this, I concluded that irrespective of the model architecture differences between corrupted and not corrupted with noise images, it remains a trivial task when the injected noise comes in big quantities (e.g., noise = 0.5).","Answers to the second batch of questions",1)
