import time
import torch.nn.init as init
import numpy as np
import os
import sys
import torchvision
from models import *
from torchvision import models, transforms


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def model_cifar(model_name, ckpt_path):
    if model_name == 'resnet':
        model = ResNet18()
    elif model_name == 'vgg':
        model = VGG('VGG16')
    elif model_name == 'densenet':
        model = DenseNet121()
    else:
        raise ValueError('Model name {} not recognized'.format(model_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Load saved weights and stats
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean=CIFAR_MEAN, std=CIFAR_STD)
    model = nn.Sequential(normalize, model)
    model = model.to(device)

    return model, best_acc


def evaluate(model, loader, uap=None, n=5):
    '''
    top         top n predicted labels (default n = 5)
    top_probs   top n probabilities (default n = 5)
    top1acc     array of true/false if true label is in top 1 prediction
    top5acc     array of true/false if true label is in top 5 prediction
    outputs     output labels
    labels      true labels
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probs, labels = [], []
    model.eval()

    if uap is not None:
        _, (x_val, y_val) = next(enumerate(loader))
        batch_size = len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])

    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):

            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.cuda()), dim=1)
            else:
                perturbed = torch.clamp((x_val.to(device) + uap).to(device), 0, 1)  # clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim=1)

            probs.append(out.cpu().numpy())
            labels.append(y_val)

    # Convert batches to single numpy arrays
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])

    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, n, axis=1)[:, :n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float16)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis=1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis=1)]

    return top, top_probs, top1acc, top5acc, outputs, labels


# dataloader for CIFAR-10
def loader_cifar(dir_data, train = False, batch_size = 250):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train:
        trainset = torchvision.datasets.CIFAR10(root = dir_data, train = True, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
    else:
        testset = torchvision.datasets.CIFAR10(root = dir_data, train = False, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True)
    return dataloader


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


import shutil

# portable terminal size (works on Windows, WSL, macOS, Linux)
size = shutil.get_terminal_size(fallback=(80, 24))
term_width = size.columns
term_height = size.lines
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
