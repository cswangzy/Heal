#!/usr/bin/env python
import socket
import time
import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import argparse
import torchvision
import torch.nn.functional as F
import random
import os
import socket
import threading
import time
import struct
from util.utils import send_msg, recv_msg, time_printer,add_model, scale_model, printer_model, time_duration
import copy
from torch.autograd import Variable
from model.model_VGG_cifar import construct_VGG_cifar
from model.model_AlexNet_cifar import construct_AlexNet_cifar
from model.model_nin_cifar import construct_nin_cifar
from model.model_VGG_image import construct_VGG_image
from model.model_AlexNet_image import construct_AlexNet_image
from model.model_nin_image import construct_nin_image
from model.model_LeNet_emnist import construct_lenet_emnist
from model.model_VGG9_cifar import construct_VGG9_cifar
from util.utils import printer
from alg import layer_selection_generation, heals_algorithm, hierfavg_algorithm,hfl_noniid,hfel,Heals_random
import math
import numpy.ma as ma
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--device_num', type=int, default=1, metavar='N',
                        help='number of working devices ')
parser.add_argument('--edge_number', type=int, default=1, metavar='N',
                        help='edge server')
parser.add_argument('--model_type', type=str, default='NIN', metavar='N',          #NIN,AlexNet,VGG
                        help='model type')
parser.add_argument('--dataset_type', type=str, default='cifar10', metavar='N',  #cifar10,cifar100,image
                        help='dataset type')
parser.add_argument('--alg_type', type=int, default='0', metavar='N',  #0-our 1-hfl 2_resource 3_label 4_random
                        help=' ip port')  
args = parser.parse_args()

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)
device_gpu = torch.device("cuda" if True else "cpu")
   
lr = 0.01
device_num = args.device_num
edge_num = args.edge_number
model_length = 0
delay_gap = 10
epoch_max = 500
epoch_div = 5
acc_count = []
criterion = nn.NLLLoss()

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 50010))
#listening_sock.bind(('172.16.50.10', 50010))

listening_sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock1.bind(('localhost', 50100))
#listening_sock.bind(('172.16.50.10', 50011))

device_sock_all = []
edge_sock_all = []

while len(edge_sock_all) < edge_num:
    listening_sock1.listen(edge_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock1.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    edge_sock_all.append(client_sock)

device_sock_all = [None]*device_num
#connect to device
for i in range(device_num):
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    msg = recv_msg(client_sock)
    print('Got connection from node '+ str(msg[1]))
    print(client_sock)
    device_sock_all[msg[1]] = client_sock


#get the information about edge and device



def send_msg_to_device_edge(sock_adr, msg):
    send_msg(sock_adr, msg)





#the algorithm stops when accuauracy of changed less than 2% in 10 epochs 
def train_stop():
    if len(acc_count)<11:
        return False
    max_acc = max(acc_count[len(acc_count)-10:len(acc_count)])
    min_acc = min(acc_count[len(acc_count)-10:len(acc_count)])
    if max_acc-min_acc <=0.0002:
        return True
    else:
        return False





def Get_band_grad_from_device_edge(epoch, total_delay):
    global bandwidth_device_edge
    acc_value = 0
    loss_value = 0
  #  bandwidth_device_edge = []
    gradient_device = []
    for i in range(device_num):
  #      bandwidth_device_edge.append([])
        msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER")
  #      bandwidth_device_edge[i] = msg[1]
    #    gradient_device.append([])
    #    gradient_device[i] = msg[2]
        acc_value += msg[3]
        loss_value+=msg[4]
    # a_1 = random.randint(1,10)
    bandwidth_device_edge = []
    for i in range(device_num):
        bandwidth_device_edge.append([])
        for j in range(edge_num):
            bandwidth_device_edge[i].append(random.randint(90,100))
    print(bandwidth_device_edge)

    if args.dataset_type == 'cifar10':
        label_num = 10
    elif args.dataset_type == "cifar100":
        label_num = 100
    elif args.dataset_type == 'image':
        label_num = 100
    elif args.dataset_type == 'emnist':
        label_num = 62
    #device_label_distribution = np.ones((device_num, label_num)) * (1.0 / device_num)
    device_label_distribution = np.ones((100, 30)) * ((1 - 0.1) / (30-1))
    a = [7, 7, 3, 11, 15, 23, 25, 17, 0, 5, 6, 24, 8, 17, 21, 2, 17, 9, 21, 4, 24, 5, 19, 18, 24, 17, 24, 28, 15, 19, 20, 17, 17, 19, 17, 27, 4, 27, 17, 22, 27, 26, 23, 8, 22, 12, 18, 18, 13, 7, 24, 14, 14, 13, 6, 18, 29, 19, 21, 27, 25, 17, 18, 25, 18, 16, 9, 11, 11, 8, 0, 16, 10, 8, 13, 12, 9, 25, 22, 5, 1, 24, 1, 1, 20, 13, 22, 10, 26, 24, 22, 13, 24, 6, 13, 8, 10, 27, 16, 22]
    for i in range(len(a)):
        device_label_distribution[i][a[i]] = 0.1

    device_label_distribution = device_label_distribution.transpose()
    print(device_label_distribution)

    if args.alg_type == 0:
        edge_assign,layer_selection, per_epoch_delay = heals_algorithm(device_label_distribution,bandwidth_device_edge,args.model_type,args.dataset_type, model_length)
    elif args.alg_type == 1:
        edge_assign, layer_selection, per_epoch_delay = hierfavg_algorithm(bandwidth_device_edge,device_num, edge_num,model_length, args.model_type)
    elif args.alg_type == 2:
        edge_assign, layer_selection, per_epoch_delay = hfel(bandwidth_device_edge,model_length, args.model_type)
    elif args.alg_type == 3:
        edge_assign, layer_selection, per_epoch_delay = hfl_noniid(bandwidth_device_edge,device_label_distribution,device_num, edge_num,args.dataset_type, model_length, args.model_type)
    elif args.alg_type == 4:
        edge_assign, layer_selection, per_epoch_delay = Heals_random(bandwidth_device_edge,edge_num,device_num,model_length, args.model_type)

    # end_time = time.time()
    # a, b = time_duration(start_time, end_time)
    printer("Epoch {} Duration {}s Testing loss: {} Testing_acc: {}".format(epoch,total_delay,loss_value/device_num,acc_value/device_num))
    return edge_assign, layer_selection, per_epoch_delay


bandwidth_edge = []
for i in range(edge_num):
    bandwidth_edge.append([])
    bandwidth_edge[i].append(55+ random.uniform(0,0.001))
print(bandwidth_edge)

def Get_band_grad_from_edge():
    global layer_size
    global bandwidth_edge
#    gradient_edge = []
    for i in range(edge_num):
        msg = recv_msg(edge_sock_all[i],"CLIENT_TO_SERVER") 
        # bandwidth_edge.append(msg[1])
        # gradient_edge.append(msg[2])
    # bandwidth_edge = []
    # a_2 = random.randint(10,100)
    # for i in range(edge_num):
    #     bandwidth_edge.append([])
    #     bandwidth_edge[i].append(a_2+random.uniform(0,0.0001))
    edge_id = []
    for i in range(edge_num):
        edge_id.append(i)
    if args.alg_type == 0:
        layer_selection,_ = layer_selection_generation(edge_id, bandwidth_edge, 0, args.model_type)
    else:
        layer_selection = np.ones((edge_num, model_length), dtype = np.int)
    max_delay = 0
    for i in range(edge_num):
        size = 0
        for j in range(model_length):
            size = size + layer_selection[i][j] * layer_size[j]
        delay = size/bandwidth_edge[i][0]
        if max_delay<delay:
            max_delay = delay
    return layer_selection, max_delay

def rev_msg_edge(sock,epoch):
    global rec_models
    global rec_time
    msg = recv_msg(sock,"CLIENT_TO_SERVER")
    rec_models.append(msg[1])
    rec_layers.append(msg[2])


rec_models = []
rec_layers = []
flow_count = 0
if args.dataset_type == "image":
    if args.model_type == "NIN":
        print(1)
        models, optimizers = construct_nin_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 16
        print(2)
        layer_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_image([0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_image([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
        layer_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
else:
    if args.model_type == "NIN":
        print(1)
        models, optimizers = construct_nin_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        print(2)
        model_length = 12
    elif args.model_type == "AlexNet":
        models, optimizers = construct_AlexNet_cifar([0,0,0,0,0,0,0,0,0,0,0],lr)
        layer_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
        model_length = 11
    elif args.model_type == "VGG":
        models, optimizers = construct_VGG_cifar([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 21
        layer_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif args.model_type == 'LeNet':
        models, optimizers = construct_lenet_emnist([0,0,0,0,0,0,0],lr) 
        model_length = 7 
        layer_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]
    elif args.model_type == 'VGG9':
        models, optimizers = construct_VGG9_cifar([0,0,0,0,0,0,0,0,0,0,0,0],lr)
        model_length = 12
        layer_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]

print("over")
sum = 0
for model in models:
    model_size = 0
    count = 0
    if model!=None:
        for para in model.parameters():
            model_size+=sys.getsizeof(para.storage())/(1024*1024/8)
        print("layer " +str(count) + "model size " +str(model_size)+"Mb")
        count+=1
        sum += model_size
print("total model size"+str(sum))

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
                    -param1.data + dict_params2[name1].data)
    return dst_model

def scale_model_one(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model


def model_add_with_partition(rec_models, models_rec, node_num):
    global models
    models_rec = np.array(models_rec)
    for i in range(len(rec_models)-1):
        for j in range(len(models_rec[i])):
            if models_rec[i][j] == 1:
                rec_models[node_num][j] = copy.deepcopy(add_model_one(rec_models[node_num][j], rec_models[i][j]))
    for j in range(len(models_rec[0])):
        rec_models[node_num][j] = copy.deepcopy(minus_model_one(rec_models[node_num][j], models[j]))   
    for i in range(len(rec_models[node_num])):
        rec_models[node_num][i] = copy.deepcopy(scale_model_one(rec_models[node_num][i],1.0/(np.sum(models_rec[:,i] == 1))))
    return rec_models[node_num]

def update_flow(layer_selection):
    global models
    global flow_count
    for i in range(len(layer_selection)):      
        for model,j in zip(models,range(model_length)):
            if layer_selection[i][j] == 1:
                for para in model.parameters():
                    flow_count += sys.getsizeof(para.storage())/(1024*1024/8)

#start_time = time.time()

edge_assign = np.zeros(device_num, dtype=np.int)  
layer_selection = np.ones((device_num,model_length),dtype = np.int)
for i in range(device_num):
    edge_assign[i] = i%edge_num
print(edge_assign,layer_selection)

for i in range(edge_num):
    msg = ['SERVER_TO_CLIENT', models, edge_assign]
    send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
    send_edge_msg.start()

for i in range(device_num):
    msg = ['SERVER_TO_CLIENT', edge_assign[i], layer_selection[i]]
    send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[i], msg))
    send_device_msg.start()

total_delay = 0
start_time  = time.time()
for epoch in range(1, epoch_max):
    total_delay = time.time() - start_time

    edge_assign, layer_selection, per_epoch_delay = Get_band_grad_from_device_edge(epoch, total_delay)
    
  #  total_delay += per_epoch_delay #download is teo times faster than domnload
    print("edge assignment and layer selection " + str(edge_assign) +str(layer_selection))
    update_flow(layer_selection)
    for i in range(device_num):
        msg = ['SERVER_TO_CLIENT', edge_assign[i], layer_selection[i]]
        send_device_msg = threading.Thread(target=send_msg_to_device_edge, args=(device_sock_all[i], msg))
        send_device_msg.start()
    for i in range(edge_num):
        msg = ['SERVER_TO_CLIENT', edge_assign, layer_selection]
        send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
        send_edge_msg.start()

    if epoch % epoch_div == 0:
        layer_selection_from_edge, per_epoch_delay = Get_band_grad_from_edge()
      #  total_delay += per_epoch_delay
        total_delay = time.time() - start_time
        print("layer_selection_from_edge {}".format(layer_selection_from_edge))
        for i in range(edge_num):
            msg = ['SERVER_TO_CLIENT', layer_selection_from_edge[i]]
            send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
            send_edge_msg.start()
        rev_msg_d = []
        for i in range(edge_num):
        # msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER") #get the parameter [0,weight]
            print("rec models")
            rev_msg_d.append(threading.Thread(target = rev_msg_edge, args = (edge_sock_all[i],epoch)))
            rev_msg_d[i].start()
        for i in range(edge_num):
            rev_msg_d[i].join()

        rec_models.append(copy.deepcopy(models))
        models = copy.deepcopy(model_add_with_partition(rec_models, rec_layers, edge_num))
        # for model in models:
        #     for para in model.parameters():
        #         print(para)
        rec_models.clear()
        rec_layers.clear()
        update_flow(layer_selection_from_edge)
        for i in range(edge_num):
            msg = ['SERVER_TO_CLIENT', models, edge_assign]
            send_edge_msg = threading.Thread(target=send_msg_to_device_edge, args=(edge_sock_all[i], msg))
            send_edge_msg.start()
     #   test(models, testloader, "Test set", epoch, start_time)
    printer("flow_count "+str(2*flow_count)+'Mb')

    if train_stop():
        break

print("The traing process is over")



