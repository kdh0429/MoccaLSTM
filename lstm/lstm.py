#!/usr/bin/python3
"""
Pytorch Variational Autoendoder Network Implementation
"""
from itertools import chain
import time
import json
import pickle
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import wandb



class FrictionLSTM(nn.Module):
    """
    VAE, x --> mu, log_sigma_sq --> N(mu, log_sigma_sq) --> z --> x
    """

    def __init__(self, config, checkpoint_directory):
        super(FrictionLSTM, self).__init__()
        self.config = config
        
        self._device = config['device']['device']
        self.num_epochs = config.getint('training', 'n_epochs')
        self.cur_epoch = 0
        self.checkpoint_directory = checkpoint_directory
        self._save_every = config.getint('model', 'save_every')


        self.model_name = '{}{}'.format(config['model']['name'], config['model']['config_id'])
        
        lstm_structure_config_dict = {'input_size': self.config.getint("data", "n_input_feature"),
        'hidden_size': self.config.getint("model", "hidden_size"),
        'num_layers': self.config.getint("model", "num_layers"),
        'bias': self.config.getboolean("model", "bias"),
        'batch_first': self.config.getboolean("model", "batch_first"),
        'dropout': self.config.getint("model", "dropout"),
        'bidirectional': self.config.getboolean("model", "bidirectional")}
        
        self.lstm = nn.LSTM(**lstm_structure_config_dict)
        self.linear = nn.Linear(lstm_structure_config_dict["hidden_size"],self.config.getint("data", "n_output"))

        self._optim = optim.Adam(
            self.parameters(),
            lr=config.getfloat('training', 'lr'),
            betas=json.loads(config['training']['betas'])
        )

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        predictions = self.linear(lstm_out[:,-1,:])
        return predictions

    def _to_numpy(self, tensor):
        return tensor.data.cpu().numpy()

    def fit(self, trainloader, validationloader, print_every=1):
        """
        Train the neural network
        """

        for epoch in range(self.cur_epoch, self.cur_epoch + self.num_epochs):
            print("--------------------------------------------------------")
            print("Training Epoch ", epoch)
            self.cur_epoch += 1
            if epoch == self.num_epochs/2:
                self._optim = optim.Adam(
                    self.parameters(),
                    lr=self.config.getfloat('training', 'lr')/10.0,
                    betas=json.loads(self.config['training']['betas'])
                )

            # temporary storage
            train_losses = []
            batch = 0
            for inputs, outputs in trainloader:
                self.train()
                self._optim.zero_grad()
                inputs = inputs.to(self._device)
                outputs = outputs.to(self._device)
                predictions = self.forward(inputs)
                train_loss = nn.L1Loss(reduction='sum')(predictions, outputs) / inputs.shape[0]
                train_loss.backward()

                self._optim.step()

                train_losses.append(self._to_numpy(train_loss))
                batch += 1
            print('Training Loss: ', np.mean(train_losses))

            validation_loss = self.evaluate(validationloader)
            print("Validation Loss: ", np.mean(validation_loss))

            if epoch % self._save_every == 0:
                self.save_checkpoint(validation_loss)

            if self.config.getboolean("log", "wandb") is True:
                wandb_dict = dict()
                wandb_dict['Training Loss'] = np.mean(train_losses)
                wandb_dict['Validation Loss'] = validation_loss
                wandb.log(wandb_dict)

        self.calculate_threshold(validationloader)
        self.save_checkpoint(validation_loss)

    def test(self, testloader, test_collision_loader=None):
        print("--------------------------- TEST ---------------------------")
        self.eval()

        predictions = []
        test_losses = []
        residuals = []

        for inputs, outputs in testloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            test_loss = nn.L1Loss(reduction='sum')(preds, outputs) / inputs.shape[0]
            predictions.extend(self._to_numpy(preds))
            residuals.extend(np.abs(self._to_numpy(preds)-self._to_numpy(outputs)))
            test_losses.append(self._to_numpy(test_loss))

        print("Test Loss: ", np.mean(test_losses))
        np.savetxt("./result/testing_result.csv", predictions, delimiter=",")
        
        collision_pred = torch.zeros(len(residuals))
        for idx, resi in enumerate(residuals):
            for i in range(2):
                if resi[i] > self.threshold[i]:
                    collision_pred[idx] = 1
        print("False Positive Rate: ", 100*np.sum(self._to_numpy(collision_pred)) / len(predictions))

        if test_collision_loader is not None:
            predictions = []

            for inputs, outputs in test_collision_loader:
                inputs = inputs.to(self._device)
                preds = self.forward(inputs)
                predictions.extend(self._to_numpy(preds))

            np.savetxt("./result/testing_result_collision.csv", predictions, delimiter=",")


    def evaluate(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        validation_losses = []
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            validation_loss = nn.L1Loss(reduction='sum')(preds, outputs) / inputs.shape[0]
            validation_losses.append(self._to_numpy(validation_loss))

        return np.mean(validation_losses)

    def calculate_threshold(self, validationloader):
        """
        Evaluate accuracy.
        """
        self.eval()
        self.threshold = torch.zeros(2).to(self._device)
        for inputs, outputs in validationloader:
            inputs = inputs.to(self._device)
            outputs = outputs.to(self._device)
            preds = self.forward(inputs)
            threshold_batch = torch.max(torch.abs(preds-outputs),0)[0]
            for i in range(2):
                if threshold_batch[i] > self.threshold[i]:
                    self.threshold[i] = threshold_batch[i]
        print("Threshold: ", self.threshold)

    def save_checkpoint(self, val_loss):
        """Save model paramers under config['model_path']"""
        model_path = '{}/epoch_{}-f1_{}.pt'.format(
            self.checkpoint_directory,
            self.cur_epoch,
            val_loss)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self._optim.state_dict()
        }
        torch.save(checkpoint, model_path)

    def restore_model(self, epoch):
        """
        Retore the model parameters
        """
        model_path = '{}{}_{}.pt'.format(
            self.config['paths']['checkpoints_directory'],
            self.model_name,
            epoch)
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self._optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cur_epoch = epoch
