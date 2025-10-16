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
        @param shape: list with size of each layer
        @param activations: type of activation function for each layer (all neurons in a layer use same activation)
        @param lr: learning rate
        @loss_fn: loss function to use
        @optim: optimization algorithm to use
        """
        super().__init__()
        self.is_binary = shape[-1] == 1
        self.device = get_torch_device()
        # create list of layers
        layer_list = [nn.Linear(x, y) for x, y in zip(shape[:-1], shape[1:])]
        # create list of activations
        activations = [act_dict[a]() for a in activations]
        # create list of layers alternating with activations: [layer1, act1, ..., layerN, actN]
        layer_act_list = list(itertools.chain.from_iterable(zip(layer_list, activations)))
        layer_act_list.insert(2, nn.Dropout())
        # create sequential module. It chains all the layers and activations sequentially
        self.model = nn.Sequential(*layer_act_list).to(self.device)
        if loss_fn is None:
            self.loss_fn = F.binary_cross_entropy if self.is_binary else F.cross_entropy
        else:
            self.loss_fn = loss_fn
        # initialize optimizer
        self.optimizer_type = optimizer

    def forward(self, x):
        """ implement forward method by calling the forward method of the Sequential module self.model """
        return self.model(x)

    def _pred(self, outputs):
        """ transform model's outputs into binary predictions (if model is binary classifier)
            or index of highest probability class (argmax) otherwise """
        if self.is_binary:
            return (outputs >= 0.5) * 1.
        else:
            return torch.argmax(outputs, dim=1)

    def predict(self, x):
        """ predict labels. 0 or 1 if binary, top class (argmax) if not """
        return self._pred(self(x))

    def fit(self, trainloader, valloader, lr, epochs, val_interval=10):
        """ fit (train) the model
        @param trainloader: training set in form of dataloader
        @param valloader: validation set in form of dataloader
        @param epochs: number of iterations over the dataset to train
        @param val_interval: interval between output logs
        @return: loss and accuracy history
        """
        loss_history = {'train': [], 'val': []}
        acc_history = {'train': [], 'val': []}
        optimizer = self.optimizer_type(self.parameters(), lr=lr, weight_decay=1e-5)
        self.train()
        with tqdm(range(epochs), desc=f'Training DNN model with lr {lr}', file=sys.stdout) as pbar:
            val_acc, val_loss = 0., 0.
            for epoch in pbar:
                accuracy, running_loss = 0., 0.
                for x_batch, y_batch in trainloader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    # get predictions for the batch
                    pred = self(x_batch)
                    # compute loss
                    loss = self.loss_fn(pred, y_batch)
                    # backpropagate the loss and compute gradients using autograd
                    loss.backward()
                    # perform optimization step (update weights with computed gradients)
                    optimizer.step()
                    # zero-out gradients before next iteration
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    accuracy += (self._pred(pred) == y_batch).sum().item() / x_batch.shape[0]

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
        correct, total, loss = 0., 0., 0.
        # we don't want to compute any gradiends here, we're just testing
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in testloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predict outputs
                pred = self(x_batch)
                # compute loss if compute_loss==True
                if compute_loss:
                    loss += self.loss_fn(pred, y_batch)
                # get number of correct predictions
                correct += (self._pred(pred) == y_batch).sum().item()
                total += x_batch.shape[0]
        # return accuracy and loss if requested, otherwise only accuracy
        accuracy = correct/total
        if compute_loss:
            return accuracy, (loss/total).item()
        return accuracy
