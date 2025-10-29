import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import argparse

from uap_incomplete import gen_universal_adv_perturbation
from utilities import loader_cifar, model_cifar, evaluate

parser = argparse.ArgumentParser(description='Attack and evaluation')
parser.add_argument('--model_name', default='resnet', type=str, help='model name')
parser.add_argument('--ckname', default='resnet', type=str, help='model checkpoint name')
parser.add_argument('--perturbation_name', default='perturbation', type=str, help='perturbation name')
parser.add_argument('--evaluate_perturbation', '-e', action='store_true')
parser.add_argument('--epochs', default=10, type=int, help='attack epochs')
parser.add_argument('--epsilon', default=0.04, type=float, help='epsilon value')
parser.add_argument('--beta', default=10, type=int, help='beta value')


args = parser.parse_args()

dir_data = 'data/cifar10'
dir_uap = 'uaps/'

testloader = loader_cifar(dir_data = dir_data, train = False)

model, best_acc = model_cifar(args.model_name, ckpt_path = 'checkpoint/'+args.ckname+'.pth')
print("The current checkpoint accuracy on the test set is: ", best_acc)


if args.evaluate_perturbation:
    uap = torch.load('uaps/' + args.perturbation_name + '.pth', map_location=torch.device('cpu'))
    _, _, _, _, outputs, labels = evaluate(model, testloader, uap=uap)
    print('After attack accuracy:', sum(outputs == labels) / len(labels))

else:
    nb_epoch = args.epochs
    eps = args.epsilon
    beta = args.beta
    uap, losses = gen_universal_adv_perturbation(model, testloader, nb_epoch, eps, beta)

    #save the perturbation matrix
    torch.save(uap, 'uaps/'+args.perturbation_name+'.pth')
    # plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))
    # plt.figure.savefig('uaps/uap3.png')

    # evaluate
    _, _, _, _, outputs, labels = evaluate(model, testloader, uap=uap)
    print('Accuracy:', sum(outputs == labels) * 100/ len(labels) )
