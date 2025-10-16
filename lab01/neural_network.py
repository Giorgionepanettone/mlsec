import itertools
import sys

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

act_dict = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'prelu':nn.PReLU, 'identity': nn.Identity, 'sigmoid': nn.Sigmoid}

__torch_device = None

def get_torch_device():
    global __torch_device

    if __torch_device is not None:
        return __torch_device

    if  torch.backends.mps.is_available():
        print('Using MPS device for hardware acceleration')
        __torch_device = 'mps'
    elif torch.cuda.is_available():
        print('Using CUDA device for hardware acceleration')
        __torch_device = 'cuda'
    else:
        print('Hardware acceleration not available, using CPU')
        __torch_device = 'cpu'
    return __torch_device


class NeuralNetwork(nn.Module):
    def __init__(self, shape, activations, loss_fn=None, optimizer=torch.optim.SGD):
        """
        @param shape: iterable with dimensions of each layer
        @param activations: type of activation function for each layer (all neurons in a layer use same activation)
        @loss_fn: loss function to use during training
        @optim: optimizer object to use for training
        """
        super().__init__()
        self.is_binary = shape[-1] == 1
        self.device = get_torch_device()
        self.optimizer_type = optimizer
        # create architecture based on shape and activations. Suggestion: see Sequential module
        # TODO
        # If loss_fn is None, setup loss function based on task (binary vs multiclass), otherwise use loss_fn
        # TODO


    def forward(self, x):
        """ implement forward pass method """
        return self.model()
        # TODO

    def _pred(self, outputs):
        """ Helper function. Transform model's outputs into binary predictions (if model is binary classifier)
            or index of highest probability class (argmax) otherwise """
        # TODO

    def predict(self, x):
        """ predict labels. 0 or 1 if binary, top class (argmax) otherwise """
        # TODO

    def fit(self, trainloader, valloader, lr, epochs, val_interval=10):
        """ fit (train) the model
        @param trainloader: training set in form of dataloader
        @param valloader: validation set in form of dataloader
        @param lr: learning rate
        @param epochs: number of training iterations over the dataset
        @param val_interval: epochs interval between evaluations on validation data

        @return: loss and accuracy history for training and validation data
        """
        loss_history = {'train': [], 'val': []}
        acc_history = {'train': [], 'val': []}
        self.train()
        # pretty progress bar
        with tqdm(range(epochs), desc=f'Training DNN model with lr {lr}', file=sys.stdout) as pbar:
            val_acc, val_loss = 0., 0.
            for epoch in pbar:
                accuracy, running_loss = 0., 0.
                # iterate over all minibatches and update parameters
                for ...
                    # TODO
                    running_loss += loss.item()
                    accuracy += (self._pred(pred) == y_batch).sum().item() / x_batch.shape[0]

                # printing and history tracking
                train_acc = accuracy / len(trainloader)
                train_loss = running_loss/len(trainloader)
                loss_history['train'].append((epoch, train_loss))
                acc_history['train'].append((epoch, train_acc))
                if (epoch + 1) % val_interval == 0:
                    val_acc, val_loss = self.test(valloader, compute_loss=True)
                    loss_history['val'].append((epoch, val_loss))
                    acc_history['val'].append((epoch, val_acc))
                pbar.set_postfix({'Train acc': train_acc,
                                  'Train loss': train_loss,
                                  'Val acc': val_acc,
                                  'Val loss': val_loss})
        return {'Loss': loss_history, 'Accuracy': acc_history}

    def test(self, testloader, compute_loss=False):
        """ Test the model.
        @param testloader: test set in form of dataloader
        @param compute_loss: whether to compute loss or not

        @return: If compute loss is True, a tuple (accuracy, loss), otherwise only accuracy
        """
        correct, total, loss = 0., 0., 0.
        self.eval()
        # we don't want to compute any gradiends here, we're just testing
        with torch.no_grad():
            # iterate over test minibatches and compute accuracy
            for ...
                # TODO
        # return accuracy and loss if requested, otherwise only accuracy
        accuracy = correct/total
        if compute_loss:
            # use item to return a python scalar rather than tensor
            return accuracy, (loss/total).item()
        return accuracy
