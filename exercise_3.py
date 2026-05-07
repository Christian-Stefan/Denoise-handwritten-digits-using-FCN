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
### ---- End General Imports --- ###

class Fit_Predict():
    def __init__(self, batch_size, mode:str,
                 data_loc=r"/DataSets", weights_path:str=None):
        """
        Def:
        --- Constructor Class argumnets ---
        :parma ... model:
        :param ... optimizer:...
        """
        self.model_label:str = mode
        self.mode = mode
        self.weights_path = weights_path

        # 1. Data loading...
        self.train_loader, self.test_loader, self.val_loader = self.pipeline(batch_size, data_loc)

    def train(self, epochs):
        """
        Def:
        :param
        :param
        """

        # 1. Loading a 'mode'
        if self.mode in ['train_denoiser','train_detector','adversial', 'training_Super_detector', 'training_Super_denoiser']:

            # 
            # 1.1 Denoiser mode ~ To train a network that can reconstruct not blurred representation of the digits you need 
            # ...'MSE' and different activation functions 
            # ... a certain width wrapping up the number of neurons one layer should hold;
            # ... a certain network (e.g., BasicFCN)
            if self.mode == 'train_denoiser': 
                print("Loading denoiser...")
                self.loss = nn.MSELoss() 
                activation = ['LeakyReLU','Tanh']
                width_complex:list = [32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2]
                self.model = BasicFCN(activation=activation, width=width_complex)
                self.optim = SGD(params=self.model.parameters(), lr=0.01)
        
            # 1.2 Detector mode ~ which differs from the previous model through its objective (which consists in 
            # distinghuishing noisy from clean images);
            # ... Header: Sigmoid which takes in one neuron;
            # ... Loss Function: BCE
            elif self.mode == 'train_detector':
                print("Loading detective...")
                self.loss = nn.BCELoss()
                activation = ['LeakyReLU','Sigmoid']
                width_complex:list = [32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]
                activation:list = ['LeakyReLU', 'Sigmoid']
                self.model = Noisy_Image_Identifier_Net(activation=activation, width=width_complex)
                self.optim = SGD(params=self.model.parameters(), lr=0.01)
        
            # 1.3. Super Detector ~ Train a detector using the denoiser as data augmentation.
            # ... Clean images are labelled as 1.
            # ... Noisy or denoised-noisy images are labelled as 0. 
            # ...   The detector is trained from scratch. The denoiser is loaded and frozen.
            elif self.mode == 'training_Super_detector':
                print("Loading detector trained with denoiser augmentation...")
                self.model_label = 'training_Super_detector'
                return self.activate_Super_Training_Detector_Mode(epochs)
            
            # 1.4 Super denoiser
            elif self.mode == 'training_Super_denoiser':
                print("Loading denoiser trained against the Super Detector...")
                self.model_label = 'training_Super_denoiser'

                return self.activate_Super_Training_Denoiser_Mode(
                    epochs=epochs,
                    alpha=0.1
                )

            # 1.5 Adversial Mode ~ Load the best version of your denoiser and the best version of your detector from
            # ... the previous exercises. Apply the denoiser on the test set to create a list of denoised
            # ... images. Then, apply your detector on the clean images, the noisy images, and the
            # ... denoised images. 
            else:
                if self.weights_path == None:
                    # 1.4 Two training session will come about if the weight's does not exist
                    self._activate_Individual_Training(epochs)

                    # 1.5 Defining the model's backbone parameters and load the weights
                    detector, denoiser, report = self.activate_Adversial_Mode(path=[
                        rf'train_detector\train_detector\train_detectormodel_weights_epoch{epochs-1}.pth',
                        rf'train_denoiser\train_denoiser\train_denoisermodel_weights_epoch{epochs-1}.pth'
                    ])
                    
                    # 1.6 Reset the label to 'adversial' to be able to adjust the training loop adequately;
                    self.model_label= 'adversial'

                    return detector, denoiser
                
                else:
                    print("Please decleare the correct path that leads to the location where the weights " \
                    "of the both, denoiser and detector models were stored")
                         
        # elif model not in [list of models]:
        #     then recover

        else:
            print("The introduced mode does not exist in the list of available modes\n" \
            "Introduce only one of the following modes:\n" \
            "->train_denoiser: Training the denoiser\n" \
            "->train_detective: Training the detective\n" \
            "->adversail: Training both, denoiser vs detective")
        
        # 2. Creating the necessary directories to store model's 'history' (e.g., losses and weights)
        self._utils_Directory_Exec()
        print("Printing the loaded model summary\n {}".format(self.model))

        # 3. Training & Validation kick-off
        # 3.1 Creating containers holding history of the model such as losses or weights;
        loss_container_train, loss_container_te = [], []
        for epoch in range(epochs):
            _loss_per_epoch_tr:float = 0.0
            _loss_per_epoch_te:float = 0.0
            #TODO Optimize the looping mechanism for faster iteration;
            # 3. Training kicks off
            self.model.train()
            for cl, ns, label in self.train_loader:
            
                # 3.1 Selecting the data assemblying mechanism based upon the mode;
                # ...for train_detective batches holding clean and noisy images are being merged;
                if self.mode == 'train_detector':
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
                    ns = conct_img_train
                    cl = conct_labels_train

                # 5. Forward pass and loss computation
                self.optim.zero_grad()
                output = self.model(ns)
                # 6. Quantifying the quality of the output; Backpropagate; Optimizer step
                _loss_tr = self.loss(output, cl) # Compare the batch of recon against the batch with clear
                _loss_tr.backward()
                self.optim.step()
                # 7. Keep track of performance
                _loss_per_epoch_tr+=_loss_tr.item()

            loss_container_train.append(_loss_per_epoch_tr/len(self.train_loader))

            # 4. Testing kicks off
            self.model.eval()
            with torch.no_grad():
                for cl, ns, label in self.test_loader:

                    if self.mode == 'train_detector':
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

                        # 4.4 Assigning old variables with concatenated batches to preserve the code taxonomy;
                        ns = conct_img_test
                        cl = conct_labels_test

                    # 6. Quantifying the quality of the output; Backpropagate; Optimizer step
                    output = self.model(ns)
                    _loss_te = self.loss(output, cl)
                    _loss_per_epoch_te+=_loss_te.item()

                    if self.model_label == 'train_detector':
                        # 7. Empirical (e.g., only for observation purposes) validation metric
                        # 7.1 Put in place the empirical validation criteiron (e.g., Counting the amount of correct instances the network classified)
                        criterion = torch.where(output>0.5, 1.0, 0.0)
                        # 7.2 Count the number of correctly predicted images on batch
                        correct_images += (criterion==conct_labels_test).sum()    
                        accuracy = (correct_images.item() / (len(self.val_loader.dataset) * 2)) * 100 
                        message = f"Correctly predicted images in % {accuracy}"
                    else:
                        message = f'Counting correctly predicted images not applicable under mode {self.model_label}'
                
                loss_container_te.append(_loss_per_epoch_te/len(self.val_loader))
            print("Epoch {} yielded: Train Loss {:.5f} | Test Loss {:.5f} | {}".format(
                epoch, loss_container_train[epoch], loss_container_te[epoch], message))
            
            # 8. Ensures the weights are saved one epoch at a time;
            torch.save(self.model.state_dict(), f'{self.model_label}\{self.model_label}\{self.model_label}model_weights_epoch{epoch}.pth')
        # 9. Ensures both containers are being saved under `high-capacity_accuracy_container\history.csv`
        # 9.1 Convert the lists to a pandas dataframe for easier saving
        raw_history_frame = pd.DataFrame({
            "train_loss": loss_container_train, 
               "test_loss": loss_container_te
        })
        # 10. Save the dataframe as a .csv file
        raw_history_frame.to_csv(f"{self.model_label}\{self.model_label}_accuracy_container\history.csv")

    def activate_Super_Training_Denoiser_Mode(self,epochs,alpha=0.1,detector_path=None,denoiser_path=None):
        """
        Train a denoiser against the frozen Super Detector.

        Objective:
            total detector_loss:        total_loss = reconstruction_loss - alpha * detector_loss
            BCE loss of the frozen detector when denoised images are labelled as noisy.
            Since we subtract this term, the denoiser is trained to increase detector loss,
            therefore fooling the detector.

        Only the denoiser parameters are updated.
        """

        # 1. Defining directories and default paths;
        self._utils_Directory_Exec()
        if detector_path is None:
            detector_path = r'training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth'
        if denoiser_path is None:
            denoiser_path = r'train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth'

        # 2. Load the paths...
        print("Loading frozen Super Detector...")
        detector = Noisy_Image_Identifier_Net(
            activation=['LeakyReLU', 'Sigmoid'],
            width=[32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]
        )
        detector.load_state_dict(torch.load(detector_path))
        detector.eval()
        for parameter in detector.parameters():
            parameter.requires_grad = False
        print("Loading denoiser to be improved...")
        denoiser = BasicFCN(
            activation=['LeakyReLU', 'Tanh'],
            width=[32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2]
        )
        denoiser.load_state_dict(torch.load(denoiser_path))

        # 3. Training setup
        self.model = denoiser
        reconstruction_loss_fn = nn.MSELoss()
        detector_loss_fn = nn.BCELoss()
        self.optim = SGD(params=denoiser.parameters(), lr=0.01)
        train_reconstruction_losses, train_detector_losses,train_total_losses = [], [], []
        test_reconstruction_losses, test_detector_losses, test_total_losses =[], [], []
        print("Printing the loaded Super Denoiser summary\n {}".format(denoiser))

        # 4. Training loop
        for epoch in range(epochs):
            denoiser.train()
            detector.eval()
            epoch_reconstruction_loss = 0.0
            epoch_detector_loss = 0.0
            epoch_total_loss = 0.0
            for cl, ns, label in self.train_loader:
                denoised_images = denoiser(ns)
                reconstruction_loss = reconstruction_loss_fn(
                    denoised_images,
                    cl
                )
                # ------------------
                # 4.1 Detector loss|
                #
                # Denoised noisy images originate from noisy images,|
                # ... so target is 0 from the detector's perspective.|
                # ... But we subtract this loss, forcing the denoiser|
                # ... to make detector loss go up.|
                # -----------------------------------------
                detector_output = detector(denoised_images)
                noisy_targets = torch.zeros_like(detector_output)
                detector_loss = detector_loss_fn(
                    detector_output,
                    noisy_targets
                )
                # 5 Combined adversarial denoiser objective
                total_loss = reconstruction_loss - alpha * detector_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                epoch_reconstruction_loss += reconstruction_loss.item()
                epoch_detector_loss += detector_loss.item()
                epoch_total_loss += total_loss.item()

            train_reconstruction_losses.append(epoch_reconstruction_loss / len(self.train_loader))
            train_detector_losses.append(epoch_detector_loss / len(self.train_loader))
            train_total_losses.append(epoch_total_loss / len(self.train_loader))
            denoiser.eval()
            detector.eval()
            epoch_reconstruction_loss_te = 0.0
            epoch_detector_loss_te = 0.0
            epoch_total_loss_te = 0.0

            with torch.no_grad():

                for cl, ns, label in self.test_loader:
                    denoised_images = denoiser(ns)
                    reconstruction_loss_te = reconstruction_loss_fn(
                        denoised_images,
                        cl
                    )
                    detector_output_te = detector(denoised_images)
                    noisy_targets_te = torch.zeros_like(detector_output_te)
                    detector_loss_te = detector_loss_fn(
                        detector_output_te,
                        noisy_targets_te
                    )
                    total_loss_te = reconstruction_loss_te - alpha * detector_loss_te
                    epoch_reconstruction_loss_te += reconstruction_loss_te.item()
                    epoch_detector_loss_te += detector_loss_te.item()
                    epoch_total_loss_te += total_loss_te.item()

            test_reconstruction_losses.append(epoch_reconstruction_loss_te / len(self.test_loader))
            test_detector_losses.append(epoch_detector_loss_te / len(self.test_loader))
            test_total_losses.append(epoch_total_loss_te / len(self.test_loader))

            print(
                "Epoch {} yielded: "
                "Train Reconstruction Loss {:.5f} | "
                "Train Detector Loss {:.5f} | "
                "Train Total Loss {:.5f} | "
                "Test Reconstruction Loss {:.5f} | "
                "Test Detector Loss {:.5f} | "
                "Test Total Loss {:.5f}".format(
                    epoch,
                    train_reconstruction_losses[epoch],
                    train_detector_losses[epoch],
                    train_total_losses[epoch],
                    test_reconstruction_losses[epoch],
                    test_detector_losses[epoch],
                    test_total_losses[epoch]
                )
            )

            torch.save(
                denoiser.state_dict(),
                rf'{self.model_label}\{self.model_label}\{self.model_label}model_weights_epoch{epoch}.pth'
            )
        raw_history_frame = pd.DataFrame({
            "train_reconstruction_loss": train_reconstruction_losses,
            "train_detector_loss": train_detector_losses,
            "train_total_loss": train_total_losses,
            "test_reconstruction_loss": test_reconstruction_losses,
            "test_detector_loss": test_detector_losses,
            "test_total_loss": test_total_losses,
            "alpha": [alpha] * epochs
        })

        raw_history_frame.to_csv(
            rf"{self.model_label}\{self.model_label}_accuracy_container\history.csv"
        )

        # 6. Plot
        PLT.figure(figsize=(8, 5))

        PLT.plot(train_reconstruction_losses, label="Train reconstruction loss")
        PLT.plot(train_detector_losses, label="Train detector loss")
        PLT.plot(test_reconstruction_losses, label="Test reconstruction loss")
        PLT.plot(test_detector_losses, label="Test detector loss")

        PLT.xlabel("Epoch")
        PLT.ylabel("Loss")
        PLT.title("Super Denoiser training: reconstruction loss and detector loss")
        PLT.legend()
        PLT.grid(alpha=0.3)
        PLT.tight_layout()
        PLT.show()

        return denoiser, detector

    def activate_Super_Training_Detector_Mode(self, epochs):
        # 11. Creates directories & initialize backbone architecture
        self._utils_Directory_Exec()
        detector = Noisy_Image_Identifier_Net(
            activation=['LeakyReLU', 'Sigmoid'],
            width=[32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]
        )
        # 12 Load the pre-trained denoiser;
        denoiser = BasicFCN(
            activation=['LeakyReLU', 'Tanh'],
            width=[32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2]
        )
        print("Loading pretrained denoiser...")
        denoiser_path = rf'train_denoiser_3\train_denoiser\train_denoisermodel_weights_epoch{19}.pth'
        denoiser.load_state_dict(torch.load(denoiser_path))
        denoiser.eval()

        # 13. Ensuring no updates in denoiser parameters take place
        for parameter in denoiser.parameters():
            parameter.requires_grad = False

        # 14. Training setup
        self.model = detector
        self.loss = nn.BCELoss()
        self.optim = SGD(params=detector.parameters(), lr=0.01)
        loss_container_train:list = []
        loss_container_te:list = []
    
        
        
        # 14.1 Repeated steps:
        # ... Init container to store denoised, noisy and clean images
        clean_images_container:list = []
        noisy_images_container:list = []

        denoised_images_container:list = []
        clean_likelihood_container:list = []
        noisy_likelihood_container:list = []
        denoised_likelihood_container:list = []
        print("Printing the loaded detector summary with label {}\n {}".format(self.model_label,detector))
                
        for epoch in range(epochs):
            # 14.1 Training starts
            detector.train()
            _loss_per_epoch_tr:float = 0.0
            for cl, ns, label in self.train_loader:
                current_b_size = cl.shape[0] # retriving batch size
                # 14.2 Feeding denoiser with ns ~ batch holding noisy images
                with torch.no_grad():
                    denoised_ns = denoiser(ns)
                # 14.3 Random choice 50% noisy, 50% denoised
                # ... generates a one-dimensional tensor holding randomly generated numbers (0,1)
                # ... then by comparison < 0.5 those numbers will be converted in boolean Flags (True/False)
                # ... then use boolean indexing to create a mixture of noisy-authentic images and reconstruction images
                use_denoised_mask = torch.rand(current_b_size) < 0.5
                augmented_noisy = ns.clone()
                augmented_noisy[use_denoised_mask] = denoised_ns[use_denoised_mask]

                # 14.4 Build detector batch; Shuffle batch
                # ... Clean images -> label 1
                # ... Noisy/denoised-noisy images -> label 0
                detector_inputs = torch.cat((cl, augmented_noisy), dim=0)
                clean_labels = torch.ones(current_b_size, 1)
                noisy_labels = torch.zeros(current_b_size, 1)
                detector_labels = torch.cat((clean_labels, noisy_labels), dim=0)
                shuffle_indices = torch.randperm(detector_inputs.size(0))
                detector_inputs = detector_inputs[shuffle_indices]
                detector_labels = detector_labels[shuffle_indices]

                # 14.5 Forward; Backpropagation; 
                self.optim.zero_grad()
                output = detector(detector_inputs)
                _loss_tr = self.loss(output, detector_labels)
                _loss_tr.backward()
                self.optim.step()
                _loss_per_epoch_tr += _loss_tr.item()
            loss_container_train.append(_loss_per_epoch_tr / len(self.train_loader))
            # 15. Validation starts;
            detector.eval()
            _loss_per_epoch_te, correct_images, total_images = 0.0, 0, 0
            with torch.no_grad():
                for cl, ns, label in self.test_loader:
                    # Repetition of the same steps as before
                    current_b_size = cl.shape[0]
                    denoised_ns = denoiser(ns)
                    use_denoised_mask = torch.rand(current_b_size) < 0.5
                    augmented_noisy = ns.clone()
                    augmented_noisy[use_denoised_mask] = denoised_ns[use_denoised_mask]
                    detector_inputs = torch.cat((cl, augmented_noisy), dim=0)
                    clean_labels = torch.ones(current_b_size, 1)
                    noisy_labels = torch.zeros(current_b_size, 1)
                    detector_labels = torch.cat((clean_labels, noisy_labels), dim=0)
                    # 14.6 Storing images;__________
                    clean_likelihood = detector(cl)#|
                    noisy_likelihood = detector(ns)#|___
                    output = detector(detector_inputs)#|_____________
                    clean_images_container.append(cl.detach().cpu())#|_____________
                    noisy_images_container.append(detector_inputs.detach().cpu())#|___
                    denoised_images_container.append(detector_inputs.detach().cpu())#|
                    # 14.7. Save detector likelihoods                                |_
                    clean_likelihood_container.append(clean_likelihood.detach().cpu())#|
                    noisy_likelihood_container.append(noisy_likelihood.detach().cpu())#|
                    denoised_likelihood_container.append(output.detach().cpu())#_______|    

                    _loss_te = self.loss(output, detector_labels)
                    _loss_per_epoch_te += _loss_te.item()
                    predictions = torch.where(output > 0.5, 1.0, 0.0)
                    correct_images += (predictions == detector_labels).sum().item()
                    total_images += detector_labels.shape[0]

            loss_container_te.append(_loss_per_epoch_te / len(self.test_loader))
            accuracy = (correct_images / total_images) * 100
            print(
                "Epoch {} yielded: Train Loss {:.5f} | Test Loss {:.5f} | Correctly predicted images in % {:.2f}".format(
                    epoch,
                    loss_container_train[epoch],
                    loss_container_te[epoch],
                    accuracy
                )
            )
            torch.save(
                detector.state_dict(),
                rf'{self.model_label}\{self.model_label}\{self.model_label}model_weights_epoch{epoch}.pth'
            )
        raw_history_frame = pd.DataFrame({
            "train_loss": loss_container_train,
            "test_loss": loss_container_te
        })
        raw_history_frame.to_csv(
            rf"{self.model_label}\{self.model_label}_accuracy_container\history.csv"
        )
        return detector, denoiser

    def activate_Adversial_Mode(self, path:[str, str], baseline_change=None):
        
        # 11. Load detector backbone architectuere and its weights;
        print("Loading weights for detector...")
        detector = Noisy_Image_Identifier_Net(activation=['LeakyReLU','Sigmoid'],
                            width=[32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1])
        detector.load_state_dict(
            torch.load(path[0])
        )

        # 12. Load denoiser backbone architecture and its weights;
        print("Loading weights for denoiser...")
        denoiser = BasicFCN(activation=['LeakyReLU','Tanh'],
                            width= [32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2]) 
        denoiser.load_state_dict(
            torch.load(path[1])
        )

        # 12.1. Apply the denoiser on the test set to create a list of denoised images
        # 12.2 Switching eval mode on for both, denoise and detector;
        denoiser.eval()
        detector.eval()
        
        # 13 Init container to store denoised, noisy and clean images
        clean_images_container:list = []
        noisy_images_container:list = []

        denoised_images_container:list = []
        clean_likelihood_container:list = []
        noisy_likelihood_container:list = []
        denoised_likelihood_container:list = []

        # 14. Starting denoiser
        print("Applying the pre-trained denoiser on the test set starts...")
        with torch.no_grad():
            for cl, ns, label in self.test_loader:            
                # 14.1. Apply denoiser on noisy test images
                denoised_images = denoiser(ns)
                # 14.2 Save denoised images, clean images and noisy images;
                clean_images_container.append(cl.detach().cpu())
                noisy_images_container.append(ns.detach().cpu())
                denoised_images_container.append(denoised_images.detach().cpu())
                # 14.3. Apply detector on clean, noisy, and denoised images;
                clean_likelihood = detector(cl)
                noisy_likelihood = detector(ns)
                denoised_likelihood = detector(denoised_images)
                # 14.4. Save detector likelihoods
                clean_likelihood_container.append(clean_likelihood.detach().cpu())
                noisy_likelihood_container.append(noisy_likelihood.detach().cpu())
                denoised_likelihood_container.append(denoised_likelihood.detach().cpu())

                
        # 15. Concatenation section: Convert list of batches into one tensor per category
        denoised_images_container = torch.cat(denoised_images_container, dim=0) # Denoised      
        clean_images_container = torch.cat(clean_images_container, dim=0) # Clean
        noisy_images_container = torch.cat(noisy_images_container, dim=0) # Noisy

        clean_likelihood_container = torch.cat(clean_likelihood_container, dim=0).squeeze() # likelihood clean
        noisy_likelihood_container = torch.cat(noisy_likelihood_container, dim=0).squeeze() # likelihood noisy
        denoised_likelihood_container = torch.cat(denoised_likelihood_container, dim=0).squeeze() # likelihood denoised
        
        # Debug Statement 
        # likelihood_evolution = denoised_likelihood_container - noisy_likelihood_container   
        
        # 16. Construct the dictionary holding the report
        adversial_report = {
            "clean_images": clean_images_container,
            "noisy_images": noisy_images_container,
            "denoised_images": denoised_images_container,
            "clean": clean_likelihood_container,
            "noisy": noisy_likelihood_container,
            "denoised": denoised_likelihood_container
           # "likelihood_evolution": likelihood_evolution
        }

        # 16. Create the boxplots based on adversial_report and obtain likelihood
        likelihood_change = self.report(adversial_report, off=False, baseline_change=baseline_change)
        adversial_report["likelihood_change_noisy_space"] = likelihood_change

        return detector, denoiser, adversial_report

    def pipeline(self, batch_size, data_loc):
        """
        Def:
        :param...
        :param...
        """

        def __flattener(data:tuple):
            f = nn.Flatten(start_dim=0)
            flattened = [f(item) if isinstance(item, torch.Tensor) else item for item in data[:-1]]
            return tuple(flattened) + (data[-1],)
    
        basisc_transf = v2.Compose([
            # transforms.ToTensor(), <- Data is of tensor nature by definition
            # v2.Normalize(mean=[0.5],std=[0.5]),
            __flattener
        ])
        
        train, test, val = create_dataloaders(data_loc=data_loc, batch_size=batch_size, transform=basisc_transf)
        
        return train, test, val

    def report(self, history:list|str|dict, off=True, baseline_change=None):
        """
        Motivation:
        Create reports for model history or adversial detector likelihoods.

        :param history:
            Can be:
            - list
            - str path
            - dict containing clean/noisy/denoised likelihood tensors
        :param off:
            If True, reporting is skipped.
        """

        if off:
            return

        elif isinstance(history, dict):

            clean_scores = 1 - history["clean"].numpy()
            noisy_scores = 1 -history["noisy"].numpy()
            denoised_scores =1 -history["denoised"].numpy()
            likelihood_evolution = denoised_scores - noisy_scores

            PLT.figure(figsize=(8, 5))

            if baseline_change is None:
                PLT.boxplot(
                    [likelihood_evolution],
                    labels=["Current setup"]
                )

            elif isinstance(baseline_change, list):
                baseline_change = [
                    np.asarray(change).reshape(-1)
                    for change in baseline_change
                ]

                current_change = np.asarray(likelihood_evolution).reshape(-1)

                PLT.boxplot(
                    baseline_change + [current_change],
                    labels=[
                        "Original detector\nOriginal denoiser",
                        "Super detector\nOriginal denoiser",
                        "Super detector\nSuper denoiser"
                    ]
                )

            else:
                baseline_change = np.asarray(baseline_change).reshape(-1)
                current_change = np.asarray(likelihood_evolution).reshape(-1)

                PLT.boxplot(
                    [baseline_change, current_change],
                    labels=[
                        "Original detector",
                        "Current detector"
                    ]
                )

            PLT.axhline(0, color="black", linewidth=1)
            PLT.ylabel("Change in noisy-likelihood: denoised - noisy")
            PLT.title("Change in detector likelihood after denoising")
            PLT.grid(axis="y", alpha=0.3)
            PLT.tight_layout()
            PLT.show(block=False)
            PLT.pause(0.1)


            # 1. Extracting the images
            clean_images = history["clean_images"].detach().cpu()
            noisy_images = history["noisy_images"].detach().cpu()
            denoised_images = history["denoised_images"].detach().cpu()

            # 2. Finding the most and least fooled plus the median changes;
            # 2.1 Identifying the likelihood indicies corresponding to the targeted status
            most_fooled_idx = int(np.argmin(likelihood_evolution)) # Extracting the index that marks the location of the a somewhat noisy image that passed through 
            least_fooled_idx = int(np.argmax(likelihood_evolution))

            # 2.1.1 Calculate the median change and use it to substract from likelihood_evolution and obtain its idx;
            median_change = np.median(likelihood_evolution)
            median_idx = int(np.argmin(np.abs(likelihood_evolution - median_change)))

            # 3. Creating numerous lists in order to be able to build the 3x3 plot on looping basis
            # 3.1 Required column and row labels:
            _row_lab:list = ['Most fooled','Median change','Least fooled']
            _col_lab:list = ['Clean', 'Noisy', 'Denoised']
            # 3.2 Wrapping the indices, the images and the scores in lists     
            selected_indices:list = [most_fooled_idx, median_idx, least_fooled_idx]
            image_sets:list= [clean_images, noisy_images, denoised_images]
            score_sets:list = [clean_scores, noisy_scores,denoised_scores]

            # 4. Create the grid 
            # 4.1 Create the grid layout 3x3 of 8,8
            fig, axes = PLT.subplots(3, 3, figsize=(8, 8))
            # 4.2 Iterate through existent columns on row basis;
            for row, sample_idx in enumerate(selected_indices):
                for col in range(3):
                    ax = axes[row, col] # 4.3 Set the location (e.g., [0,0] -> [0,1] -> [0,2])
                    # 4.4 Extract (a clean, noisy or denoised image)
                    # ... and use a helper function to restore image size (e.g., (1024) -> (32,32))
                    img = self._prepare_image(image_sets[col][sample_idx])
                    # 4.5 Extract correspondent likelihood
                    likelihood = score_sets[col][sample_idx]
                    # 4.6 Display image
                    ax.imshow(img)
                    ax.set_title(
                        f"{_col_lab[col]}\nNoisy Likelihood: {likelihood:.4f}",
                        fontsize=10
                    )
                    # Do labelling for each first column;
                    if col == 0:
                        ax.set_ylabel(
                            _row_lab[row],
                            fontsize=10
                        )
            fig.suptitle(
                "Detector fooledness analysis: clean, noisy, and denoised samples",
                fontsize=14
            )
            PLT.tight_layout()
            PLT.show()

            return likelihood_evolution
    
    def _utils_Directory_Exec(self, execution:bool=True):
        if execution:
            # 1. Create and check if directories wherein history is stored exist
            # 1.1 For weights amd containers that hold MSE-loss scores
            if self.model_label != None and (f'{self.model_label}' in os.listdir()) == False:
                os.makedirs(f'{self.model_label}')
                if (self.model_label in os.listdir(f'{self.model_label}')) == False:
                    os.makedirs(f'{self.model_label}\{self.model_label}')
                    os.makedirs(f'{self.model_label}\{self.model_label}_accuracy_container')
        else:
            pass
    
    def _activate_Individual_Training(self, epochs):
        self.mode = 'train_denoiser'
        self.model_label = self.mode
        self.train(epochs)
        self.mode = 'train_detector'
        self.model_label = self.mode
        self.train(epochs)
    
    def _prepare_image(self, img_tensor):
        """
        Converts a single flattened or image-shaped tensor into 2D image format.
        Assumes 32x32 flattened images if tensor has 1024 elements.
        """

        img = img_tensor.detach().cpu().numpy()

        # Case 1: flattened image, e.g. shape: [1024]
        if img.ndim == 1:
            side = int(np.sqrt(img.shape[0]))
            img = img.reshape(side, side)

        # Case 2: image with channel dimension, e.g. [1, 32, 32]
        elif img.ndim == 3:
            img = np.squeeze(img)

        return img


# TODO Document all of the function bellow if time allows;

"""
The current models holds ReLu as non-linear activation units.
The network should be a class and its constructor `__init__` should accept an argument defining the width of the network (e.g., layer sizes) - Done

e.g., YourClassName([32**2, 24**2, 16**2, 32**2]) - Done
- 32^2 inputs (one image at a time); - Done
- two hidden layers 24^2 and 16^2; - Done
- 32^2 outputs(the reconstruction of the image) - Done
"""

class BasicFCN(nn.Module):
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
        self.activation4 = hid_actv_function(0.01)
        self.btcnorm4 = nn.BatchNorm1d(width[4])
        self.fcn_hidlay4 = nn.Linear(in_features=width[4], out_features=width[5], bias=True)
        self.btcnorm5 = nn.BatchNorm1d(width[5])
        self.activation5 = hid_actv_function(0.01) # TODO Argue the reason that made you addopt `0.01`
        self.fcn_hidlay5 = nn.Linear(in_features=width[5], out_features=width[6], bias=True)
        self.activation6 = final_actv_function()
  
    def forward(self, x):
        x = self.activation1(self.btcnorm1(self.fcn_input(x)))
        x = self.activation2(self.btcnorm2(self.fcn_hidlay1(x)))
        x = self.activation3(self.btcnorm3(self.fcn_hidlay2(x)))
        x = self.activation4(self.btcnorm4(self.fcn_hidlay3(x)))
        x = self.activation5(self.btcnorm5(self.fcn_hidlay4(x)))
    
        # The Output "Head"
        x = self.fcn_hidlay5(x) 
        x = self.activation6(x) 

        return x
    

#TODO Document all of the function bellow if time allows;

"""
The current models holds ReLu as non-linear activation units.
The network should be a class and its constructor `__init__` should accept an argument defining the width of the network (e.g., layer sizes) - Done

e.g., YourClassName([32**2, 24**2, 22**2, 20**2, 22**2, 24**2, 32**2, 1]) - Done
- 32^2 inputs (one image at a time); - Done
- multiple hidden layers (24**2, 22**2, 20**2, 22**2, 24**2, 32**2) - Done
- 32^2 outputs(the reconstruction of the image) - Done
- 1 output (the classification head)
"""

class Noisy_Image_Identifier_Net(nn.Module):
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
    


# All the results (e.g., figures and plots) were stored in intuitively termed folders:
# exercise1.3_firsthalf_results -> a, b, c, d
# exercise1.3_secondhalf_results -> e,f,g
# exercise1.3_finalpart_results -> h,i.j

if __name__ == "__main__":
    pass
#     # INSTRUCTIONS:
#     # Uncomment the block(s) of code depending upon the result(s) you want to get 
#     # Keep in mind that some blocks are inextricably linked so running one block of code might mean uncommenting other

# ----------------------------------------------------- First Block ------------------------------------------------------------
#     # 1.3.a); 1.3.b); 1.3.c); 1.4.d) ---------------------------------------------------------------------------------- STARTS

#     fit = Fit_Predict(
#     batch_size=64,
#     mode="adversial",
#     data_loc=r"/DataSets"
# )

# _, _, original_report = fit.activate_Adversial_Mode(
#     path=[
#         r"Submission\exercise1.3_firsthalf_results\train_detector_submission\train_detector\train_detectormodel_weights_epoch19.pth",
#         r"Submission\exercise1.3_firsthalf_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
#     ]
# )
# original_change = original_report["likelihood_change_noisy_space"]

#     # 1.3.a); 1.3.b); 1.3.c); 1.4.d) ---------------------------------------------------------------------------------- ENDS
# ----------------------------------------------------- First Block ------------------------------------------------------------


# ----------------------------------------------------- Second Block ------------------------------------------------------------
# # In order to execute the second part of the code corresponding to the exercises enumerated bellow the "history" is required
# # ... which in this case is carried by the variable "original_change".
# # THEREFORE UNCOMMENT BOTH BLOCK OF CODES, THIS (the second block) AND THE ABOVE (the first block) in order to obtain the results for 1.3.e) 1.3.f); 1.3.g
# # 1.3.e) 1.3.f); 1.3.g);----------------------------------------------------------------------------------------------- STARTS

# _, _, super_report = fit.activate_Adversial_Mode(
#     path=[
#         r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#         r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
#     ],
#     baseline_change=original_change
# )

# # 1.3.e) 1.3.f); 1.3.g);----------------------------------------------------------------------------------------------- END
# ----------------------------------------------------- Second Block ------------------------------------------------------------

# ----------------------------------------------------- Third Block ------------------------------------------------------------
# To avoid prolonged training time please directly consult the folder ('train_denoiser_3_submission) to check out the results;
# 1.3.h); -------------------------------------------------------------------------------------------------------- START
# fit = Fit_Predict(
#     batch_size=64,
#     mode='training_Super_denoiser',
#     data_loc=r"/DataSets"
# )

# super_denoiser, frozen_detector = fit.activate_Super_Training_Denoiser_Mode(
#     epochs=20,
#     alpha=0.1,
#     detector_path=r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#     denoiser_path=r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
# )

# super_denoiser, frozen_detector = fit.activate_Super_Training_Denoiser_Mode(
#     epochs=20,
#     alpha=0.05,
#     detector_path=r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#     denoiser_path=r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
# )


# super_denoiser, frozen_detector = fit.activate_Super_Training_Denoiser_Mode(
#     epochs=20,
#     alpha=0.5,
#     detector_path=r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#     denoiser_path=r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
# )

# super_denoiser, frozen_detector = fit.activate_Super_Training_Denoiser_Mode(
#     epochs=20,
#     alpha=0.8,
#     detector_path=r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#     denoiser_path=r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
# )

# ----------------------------------------------------- Third Block ------------------------------------------------------------
#  # 1.3.j)

# _, _, super_detector_report = fit.activate_Adversial_Mode(
#     path=[
#         r"training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#         r"train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
#     ],
#     baseline_change=original_change
# )

# super_detector_change = super_detector_report["likelihood_change_noisy_space"]

# _, _, super_denoiser_report = fit.activate_Adversial_Mode(
#     path=[
#         r"training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
#         r"training_Super_denoiser\training_Super_denoiser\training_Super_denoisermodel_weights_epoch19.pth"
#     ],
#     baseline_change=[
#         original_change,
#         super_detector_change
#     ]
# )

# ----------------------------------------------------- Third Block ------------------------------------------------------------

# ----------------------------------------------------- Fourth Block ------------------------------------------------------------
#  1.3.i), 1.3.j;

    fit = Fit_Predict(
    batch_size=64,
    mode="adversial",
    data_loc=r"/DataSets"
)

_, _, original_report = fit.activate_Adversial_Mode(
    path=[
        r"Submission\exercise1.3_firsthalf_results\train_detector_submission\train_detector\train_detectormodel_weights_epoch19.pth",
        r"Submission\exercise1.3_firsthalf_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
    ]
)
original_change = original_report["likelihood_change_noisy_space"]

_, _, super_detector_report = fit.activate_Adversial_Mode(
    path=[
        r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
        r"Submission\exercise1.3_finalpart_results\train_denoiser_3_submission\train_denoiser\train_denoisermodel_weights_epoch19.pth"
    ],
    baseline_change=original_change
)

super_detector_change = super_detector_report["likelihood_change_noisy_space"]

_, _, super_denoiser_report = fit.activate_Adversial_Mode(
    path=[
        r"Submission\exercise1.3_finalpart_results\training_Super_detector_submission\training_Super_detector\training_Super_detectormodel_weights_epoch19.pth",
        r"Submission\exercise1.3_finalpart_results\training_Super_denoiser\training_Super_denoiser\training_Super_denoisermodel_weights_epoch19.pth"
    ],

    baseline_change=[
        original_change,
        super_detector_change
    ]
)

# ----------------------------------------------------- Fourth Block ------------------------------------------------------------