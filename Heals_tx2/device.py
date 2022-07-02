#!/usr/bin/env python
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import time
import struct
import argparse
from util.utils import send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from model.model_nin_emnist import construct_nin_emnist
from model.model_AlexNet_emnist import construct_AlexNet_emnist
from model.model_LeNet_emnist import construct_lenet_emnist
from model.model_VGG_emnist import construct_VGG_emnist
from model.model_VGG9_cifar import construct_VGG9_cifar
from util.utils import printer

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=3, metavar='N',
                        help='number of working devices (default: 3)')
parser.add_argument('--edge_number', type=int, default=1, metavar='N',
                        help='edge server')
parser.add_argument('--node_num', type=int, default=1, metavar='N',
                        help='device index (default: 1)')
parser.add_argument('--device_ip', type=str, default='localhost', metavar='N',
                        help=' ip address')
parser.add_argument('--device_port', type=int, default='50001', metavar='N',
                        help=' ip port')
parser.add_argument('--use_gpu', type=int, default=0, metavar='N',
                        help=' ip port')
parser.add_argument('--device_ip_list', type=list, default=['localhost'], metavar='N',
                        help=' ip port')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--use_gpu_id', type=int, default=0, metavar='N',
                        help=' ip port')  
args = parser.parse_args()

if args.use_gpu_id == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.use_gpu_id == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
elif args.use_gpu_id == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
elif args.use_gpu_id == 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
elif args.use_gpu_id == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
elif args.use_gpu_id == 5:
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
elif args.use_gpu_id == 6:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
elif args.use_gpu_id == 7:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    
# if args.use_gpu == 0:
#     print('use gpu')
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     torch.set_default_tensor_type(torch.FloatTensor)
#torch.cuda.manual_seed(args.seed) #<--random seed for one GPU
#torch.cuda.manual_seed_all(args.seed) #<--random seed for multiple GPUs
device_gpu = torch.device("cuda" if args.use_gpu == 0 else "cpu")
# Configurations are in a separate config.py file

device_num = args.device_num
node_num = args.node_num
edge_num = args.edge_number

sock_ps = socket.socket()
sock_ps.connect(('localhost', 50010))
#sock_ps.connect(('172.16.50.22', 50010))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_ps,msg)

sock_edge1 = socket.socket()
sock_edge1.connect(('localhost', 51001))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_edge1,msg)
sock_edge = []
sock_edge.append(sock_edge1)

sock_edge2 = socket.socket()
sock_edge2.connect(('localhost', 51002))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_edge2,msg)
sock_edge.append(sock_edge2)

sock_edge3 = socket.socket()
sock_edge3.connect(('localhost', 51003))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_edge3,msg)
sock_edge.append(sock_edge3)

sock_edge4 = socket.socket()
sock_edge4.connect(('localhost', 51004))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_edge4,msg)
sock_edge.append(sock_edge4)

sock_edge5 = socket.socket()
sock_edge5.connect(('localhost', 51005))
msg = ['CLIENT_TO_SERVER',node_num]
send_msg(sock_edge5,msg)
sock_edge.append(sock_edge5)

print('---------------------------------------------------------------------------')


if args.dataset_type == 'cifar100':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar-100-python/device_train_30_device/device'+str(node_num)+'/', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
   # trainset = datasets.CIFAR10('/data/zywang/Dataset/cifar10', download=True, train=True, transform=transform)
    trainset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/device_train_30_device/device'+str(node_num)+'/', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    trainset = datasets.ImageFolder('/data/zywang/Dataset/emnist/30_device_train/device'+str(node_num)+'/', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([  transforms.Resize((144,144)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    #trainset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE100/25_device_train/device'+str(node_num)+'/', transform = transform)
    trainset = datasets.ImageFolder('/data/zywang/Dataset/heals_image100_iid', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True)

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Resize((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
   # trainset = datasets.ImageFolder('/data/zywang/Dataset/image_coopfl/train', transform = transform)
    trainset = datasets.ImageFolder('/data/zywang/PartImagenet/train', transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

if args.dataset_type == 'cifar100':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar-100-python/test_cifar100', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

elif args.dataset_type == 'cifar10':
    transform = transforms.Compose([ 
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/cifar10/cifar10/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

elif args.dataset_type == 'emnist':
    transform = transforms.Compose([
                           transforms.Resize(28),
                           #transforms.CenterCrop(227),
                           transforms.Grayscale(1),
                           transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
          ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/emnist/emnist_train/byclass_test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

elif args.dataset_type == 'image' and args.model_type != "AlexNet":
    transform = transforms.Compose([
                            transforms.CenterCrop(144),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/iamge100_test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False)

elif args.dataset_type == 'image' and args.model_type == "AlexNet":
    transform = transforms.Compose([  transforms.Scale((227,227)),
                               #   transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    testset = datasets.ImageFolder('/data/zywang/Dataset/IMAGE10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

lr=0.01


criterion = nn.NLLLoss()
local_update = 10

def local_train():
    global models
    global optimizers
    print("strat training")
    for _ in range(local_update):
        count = 0
        for images, labels in trainloader:
        #    if mutex.acquire(True):
            for i in range(len(models)):
                models[i].train()
            #    mutex.release()
            count+=1
            if count % 10 == 0:
                print("batch_training "+str(count))
            # if count == 20:
            #     break
#forward
            images, labels = images.to(device_gpu), labels.to(device_gpu) 
            input[0] = images
            output[0] = models[0](input[0])
            input[1] = output[0].detach().requires_grad_()
            for i in range(1,len(models)):
                    output[i] = models[i](input[i])
                    if i<len(models)-1:
                        input[i+1] = output[i].detach().requires_grad_()
                    else:
                        loss = criterion(output[i], labels)  
                        if count%50 == 0:
                            print("trianing loss", loss) 
 #梯度初始化为zero
            for i in range(0,len(optimizers)):                         
                if optimizers[i] != None:
                    optimizers[i].zero_grad()       
#backward   
            loss.backward()
            for i in range(len(models)-2, -1, -1):
                    grad_in = input[i+1].grad
                    output[i].backward(grad_in)
#更新参数    
        #    if mutex.acquire(True):   
            for i in range(0,len(optimizers)):
                if optimizers[i] !=None:
                    optimizers[i].step()
         #       mutex.release()
    test(models, testloader, "Test set")


def test(models, dataloader, dataset_name):
    global accuracy
    global loss_value
    for model in models:
        model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            x=data.to(device_gpu)
            for i in range(0,len(models)):
                y = models[i](x)
                if i<len(models)-1:
                    x = y
                else:
                    loss += criterion(y, target)
                    pred = y.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
    print("Testing loss: {}".format(loss/len(dataloader)))
    print("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))
    accuracy = float(correct)/len(dataloader.dataset)
    loss_value = loss/len(dataloader)
    

model_length = 0
if args.dataset_type == "image":
    if args.model_type == "NIN":
        model_length = 16
    elif args.model_type == "AlexNet":
        model_length = 11
    elif args.model_type == "VGG":
        model_length = 21
else:
    if args.model_type == "NIN":
        model_length = 12
    elif args.model_type == "AlexNet":
        model_length = 11
    elif args.model_type == "VGG":
        model_length = 21
    elif args.model_type == 'LeNet':
        model_length = 7
    elif args.model_type == 'VGG9':
        model_length = 12


bandwidth_device = []
gradient_device = []
accuracy = []
loss_value = []

msg = recv_msg(sock_ps,'SERVER_TO_CLIENT')
edge_assign = msg[1]
layer_selection = msg[2]
print(edge_assign, layer_selection)

if args.dataset_type == "image":
    if args.model_type == "NIN":
        models, optimizers = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
elif args.dataset_type == 'emnist':
    if args.model_type == "NIN":
        models, optimizers = construct_nin_emnist([0,0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_emnist([0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_emnist([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == 'LeNet':
        models, optimizers = construct_lenet_emnist([0,0,0,0,0,0,0],lr) 
else:
    if args.model_type == "NIN":
        models, optimizers = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
    elif args.model_type == 'VGG9':
        models, optimizers = construct_VGG9_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)

input=[None]*len(models)
output=[None]*len(models)


while True:
    print(bandwidth_device,gradient_device)
    msg = recv_msg(sock_edge[edge_assign], 'SERVER_TO_CLIENT')
    global_model = msg[1]
 #   models, optimizers = construct_resnet(partition_way,lr)
    for i in range(len(global_model)):
        models[i] = copy.deepcopy(global_model[i])
        models[i] = models[i].to(device_gpu)
    for i in range(len(optimizers)):
        if optimizers[i]!=None:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr, momentum = 0.9)
    # for model in models:
    #     for para in model.parameters():
    #         print(para)
    local_train()
    msg = ['CLIENT_TO_SERVER',bandwidth_device, gradient_device, accuracy,loss_value]
    send_msg(sock_ps,msg) 
    msg = recv_msg(sock_ps,'SERVER_TO_CLIENT')
    edge_assign = msg[1]
    layer_selection = msg[2]
    send_models = [None]*len(models)
    # for model in models:
    #     for para in model.parameters():
    #         print(para)
    for i in range(len(models)):
        if layer_selection[i] == 1:
            send_models[i] = copy.deepcopy(models[i])

    msg = ['CLIENT_TO_SERVER', send_models, layer_selection]
    send_msg(sock_edge[edge_assign], msg)


