#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import threading
import os
import socket
import time
import struct
from util.utils import send_msg, recv_msg
import copy
import argparse
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from model.model_nin_emnist import construct_nin_emnist
from model.model_AlexNet_emnist import construct_AlexNet_emnist
from model.model_VGG_emnist import construct_VGG_emnist
from util.utils import printer, partition_way_converse, start_forward_layer, start_backward_layer, time_printer, add_model ,scale_model,printer_model
import numpy as np
from util.utils import send_msg, recv_msg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=30, metavar='N',
                        help='number of working devices ')
parser.add_argument('--edge_id', type=int, default=0, metavar='N',
                        help='edge server')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--edge_ip', type=str, default='localhost', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--edge_port', type=str, default='51001', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
args = parser.parse_args()

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device_gpu = torch.device("cuda" if True else "cpu")

device_num = args.device_num
start_time = time.time()
lr=0.01
criterion = nn.NLLLoss()
edge_label = args.edge_id  #402
epoch_div = 5
receive_model = []

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 51001+args.edge_id))

#connect to the PS
sock = socket.socket()
sock.connect(('localhost', 50100))

device_sock_all=[None]*device_num
for i in range(device_num):
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    msg = recv_msg(client_sock)
    print('Got connection from node '+ str(msg[1]))
    print(client_sock)
    device_sock_all[msg[1]] = client_sock
print("-------------------------------------------------------")
#receive partition way and construct model
rec_models = []
rec_layers = []
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
    

def test(models, dataloader, dataset_name, epoch):
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
    printer("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
    printer("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                correct,
                                                len(dataloader.dataset),
                                                100. * correct / len(dataloader.dataset)))
    

def add_model_one(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return dst_model

def minus_model_one(dst_model, src_model):
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    - param1.data + dict_params2[name1].data)
    return dst_model

def scale_model_one(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def model_add_with_partition(rec_models, models_rec, node_num):
    global global_model
    models_rec = np.array(models_rec)
    for i in range(len(rec_models)-1):
        for j in range(len(models_rec[i])):
            if models_rec[i][j] == 1:
                rec_models[node_num][j] = copy.deepcopy(add_model_one(rec_models[node_num][j], rec_models[i][j]))
    for j in range(len(models_rec[0])):
        rec_models[node_num][j] = copy.deepcopy(minus_model_one(rec_models[node_num][j], global_model[j]))
    for i in range(len(rec_models[node_num])):
        rec_models[node_num][i] = copy.deepcopy(scale_model_one(rec_models[node_num][i],1.0/(np.sum(models_rec[:,i] == 1))))
    return rec_models[node_num]


def send_msg_to_device_edge(sock_adr, msg):
    send_msg(sock_adr, msg)

def rev_msg_edge(sock,epoch):
    global rec_models
    global rec_layers
    global rec_time
    msg = recv_msg(sock,"CLIENT_TO_SERVER")
    rec_models.append(msg[1])
    rec_layers.append(msg[2])

#after connect to the PS
while True:
    msg = recv_msg(sock,'SERVER_TO_CLIENT')
    global_model = copy.deepcopy(msg[1])
    edge_assign = msg[2]
    print(edge_assign)

    for i in range(epoch_div):

        for j in range(device_num):
            if edge_assign[j]  == edge_label:
                msg = ['SERVER_TO_CLIENT', global_model]
                send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[j], msg))
                send_device_msg.start()
                print("send_model")

        msg = recv_msg(sock,'SERVER_TO_CLIENT')
        edge_assign = msg[1]
        layer_selection = msg[2]

        rev_msg_d = []
        id = 0
        for j in range(device_num):
            print("rec models")
            if edge_assign[j]  == edge_label:
                rev_msg_d.append(threading.Thread(target = rev_msg_edge, args = (device_sock_all[j],j)))
                rev_msg_d[id].start()
                id+=1
        for j in range(len(rev_msg_d)):
            rev_msg_d[j].join()
        rec_models.append(copy.deepcopy(global_model))

        global_model = copy.deepcopy(model_add_with_partition(rec_models, rec_layers, len(rec_models)-1))
        rec_models.clear()
        rec_layers.clear()

    bandwidth = []
    gradient = []
    msg = ['CLIENT_TO_SERVER',bandwidth, gradient]
    send_msg(sock,msg)
    msg = recv_msg(sock,'SERVER_TO_CLIENT')
    layer_selection_edge = msg[1]

    send_models = [None]*len(global_model)
    for i in range(len(global_model)):
        if layer_selection_edge[i] == 1:
            send_models[i] = copy.deepcopy(global_model[i])
    msg = ['CLIENT_TO_SERVER',send_models,layer_selection_edge]
    send_msg(sock,msg)

