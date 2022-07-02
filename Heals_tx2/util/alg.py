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
import copy
import math
from util.utils import minus_model, scale_model, add_model


def layer_selection_generation(neibo_id, bandwidth,node_num, str):
    #initialization, bandwidth = device*device
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]


    models_rev = np.zeros((len(neibo_id), len(model_size)), dtype=np.int )
    mu_1 = np.zeros((len(neibo_id),len(model_size)), dtype=np.float)  
    for i in range(len(mu_1)):
        for j in range(len(mu_1[i])):
            mu_1[i][j] = model_size[j]/bandwidth[neibo_id[i]][node_num]

    ef_lk = np.zeros((len(neibo_id),len(model_size)), dtype=np.float)  
    for i in range(len(ef_lk)):
        for j in range(len(ef_lk[i])):
            if np.min(mu_1[:,j]) != 0:
                ef_lk[i][j] = np.min(mu_1[:,j])/mu_1[i][j]
            else:
                ef_lk[i][j] = 1.0/math.sqrt(len(ef_lk))
    rank_device = np.zeros(len(neibo_id), dtype=np.float)  
    for i in range(len(rank_device)):
        rank_device[i] = sum(ef_lk[i])
    #mdoel aggregation
    sum_device = np.zeros(len(neibo_id), dtype=np.float) #0 exist 1 disapper
    avail_layer = np.zeros(len(model_size), dtype=np.int) 
    avail_device = np.zeros(len(neibo_id), dtype=np.int) 

    while 0 in avail_layer:
      #  print("alg6")
        device_list = copy.deepcopy(sum_device)
        k = np.argmin(device_list)
        while True:
       #     print("alg5")
            if avail_device[k] == 0 or device_list[k] == 100000000:
                break
            else:
                device_list[k] = 100000000
                k = np.argmin(device_list)
         
        layer_list = copy.deepcopy(ef_lk[k]) 
        l = np.argmax(layer_list)
        while True:
       #     print("alg4")
            if avail_layer[l] == 0 or layer_list[l] == 0:
                break
            else:
                layer_list[l] = 0
                l = np.argmax(layer_list)
                
        if ef_lk[k][l] < 1.0/math.sqrt(len(ef_lk)):
            avail_device[k] = 1
        else:
            avail_layer[l] = 1
            sum_device[k] +=  mu_1[k][l]
            models_rev[k][l] = 1
        
    max_delay = np.max(sum_device)
    for i in range(len(ef_lk)):
        for j in range(len(ef_lk[i])):
            if sum_device[i] + mu_1[i][j] <= max_delay and models_rev[i][j] == 0:
                models_rev[i][j] = 1
                sum_device[i] += mu_1[i][j]
    return models_rev, max_delay


def edge_assignment_generation(device_label_distribution, layer_selection,bandwidth,str,data_type):
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]


    if data_type == 'cifar10':
        label_num = 10
    elif data_type == "cifar100":
        label_num = 100
    elif data_type == 'image':
        label_num = 10
    elif data_type == 'emnist':
        label_num = 62
    #priority
    edge_assignment = np.zeros(len(bandwidth), dtype=np.int)
    for i in range(len(edge_assignment)):
        edge_assignment[i]=-1
    lam = np.zeros(len(bandwidth), dtype=np.float)
    tran_model = np.zeros(len(bandwidth), dtype=np.float)
    avaliable_device = np.ones(len(bandwidth), dtype=np.int)
    edk_global = np.zeros(label_num, dtype=np.float)
    edk_edge = np.zeros((len(bandwidth[1]),label_num),dtype = np.float)
    for i in range(len(edk_global)):
        edk_global[i] = float(1.0)/label_num

    for i in range(len(tran_model)):
        for j in range(len(model_size)):
            if layer_selection[i][j] == 1:
                tran_model[i] += model_size[j] 
    for i in range(len(lam)):
        lam[i] = tran_model[i]*len(bandwidth[i])/sum(bandwidth[i])
    while 1 in avaliable_device:
    #    print("alg3")
        u = np.argmax(lam)
        band_u = np.array(bandwidth[u])
        for _ in range(len(band_u)):
            v = np.argmax(band_u)
            edk_v_pre = copy.deepcopy(edk_edge[v])
            edk_v_after = copy.deepcopy(edk_edge[v])
            edk_v_pre_value = np.linalg.norm(edk_v_pre-edk_global,ord=1)
       #     print(device_label_distribution,edk_v_after,edk_v_pre)
            for i in range(len(edk_v_pre)):
                edk_v_after[i] = (edk_v_pre[i]*np.sum(edge_assignment[:]==v)+device_label_distribution[u][i])/(1+np.sum(edge_assignment[:]==v))
            edk_v_after_value = np.linalg.norm(edk_v_after-edk_global,ord=1)
            if edk_v_after_value <= edk_v_pre_value:
                edge_assignment[u] = v
                edk_edge[v] = copy.deepcopy(edk_v_after)
                avaliable_device[u] = 0
                break
            else:
                band_u[v] = 0
        if edge_assignment[u] == -1:
            edge_assignment[u] = np.argmax(bandwidth[u])
            edk_edge[edge_assignment[u]] = copy.deepcopy(edk_v_after)
            avaliable_device[u] = 0
        lam[u] = -1
    flag = True
    bandwidth = np.array(bandwidth)
    while flag:
     #   print("alg2")
        flag = False
        for i in range(len(bandwidth[0])):
            if not i in edge_assignment:
              #  edge_assignment[np.argmax(bandwidth[:,i])] = i
              edge_assignment[random.randint(0,len(edge_assignment)-1)] = i
        for i in range(len(bandwidth[0])):
            if not i in edge_assignment:
                flag  = True
    return edge_assignment

def model_divergence(w0, w1, w0_grad, w1_grad):
    new_w0 = np.array(w0[0].cpu())
    new_w1 = np.array(w1[0].cpu())
    new_w0_grad = np.array(w0_grad[0].cpu())
    new_w1_grad = np.array(w1_grad[0].cpu())
    for i in range(1,len(w0)):
        w0[i] = np.array(w0[i].cpu())
        new_w0 = np.append(new_w0,w0[i])
    for i in range(1,len(w1)):
        w1[i] = np.array(w1[i].cpu())
        new_w1 = np.append(new_w1,w1[i])
    for i in range(1,len(w0_grad)):
        w0_grad[i] = np.array(w0_grad[i].cpu())
        new_w0_grad = np.append(new_w0_grad,w0_grad[i])
    for i in range(1,len(w1_grad)):
        w1_grad[i] = np.array(w1_grad[i].cpu())
        new_w1_grad = np.append(new_w1_grad,w1_grad[i])
   # w0_grad = np.array(w0_grad)
    La = np.linalg.norm(new_w0_grad-new_w1_grad)/np.linalg.norm(new_w0-new_w1)
    epi = np.linalg.norm(new_w0_grad)
    model_divergence = np.linalg.norm(new_w0_grad)/La
    print("L=" + str(La)+ " ;epi=" +str(epi) +" ;model_divergence="+str(model_divergence))
    return La, epi, model_divergence


def edge_assignment_array_to_matric(edge_assign,edge_number):
    edge_assign_matric = []
    for i in range(edge_number):
        edge_assign_matric.append([])
    
    for i in range(len(edge_assign)):
        edge_assign_matric[edge_assign[i]].append(i)
    return edge_assign_matric

def edge_assignment_matric_to_arrary(edge_assign,device_num):
    edge_assign_array = [None]*device_num
    for i in range(len(edge_assign)):
        for j in range(len(edge_assign[i])):
            edge_assign_array[edge_assign[i][j]] = i
    return edge_assign_array


def heals_algorithm(device_label_distribution,bandwidth,model_type,data_type, model_length):
    now_delay = 10000
    edge_number = len(bandwidth[0])
    device_number = len(bandwidth)
    pre_edge_assign = np.zeros(len(bandwidth), dtype=int)  
    pre_delay = np.zeros(edge_number, dtype=float)  
    pre_layer_selection = np.zeros((device_number,model_length),dtype=int)
    for i in range(len(bandwidth)):
        pre_edge_assign[i] = i% len(bandwidth[0])
    assign_matric = edge_assignment_array_to_matric(pre_edge_assign,edge_number)
    for i in range(edge_number):
        device_id = assign_matric[i]
        layer_selection, pre_delay[i] = layer_selection_generation(device_id, bandwidth, i, model_type)
      #  print(device_id,layer_selection,pre_layer_selection)
        for j in range(len(device_id)):
            pre_layer_selection[device_id[j]] = layer_selection[j]

#     while np.max(pre_delay) < now_delay:
#    #     print("alg1")
#         now_delay = np.max(pre_delay)
#         pre_edge_assign = edge_assignment_generation(device_label_distribution, pre_layer_selection,bandwidth,model_type,data_type)
#         assign_matric = edge_assignment_array_to_matric(pre_edge_assign,edge_number)
#         for i in range(edge_number):
#             device_id = assign_matric[i]
#             layer_selection, pre_delay[i] = layer_selection_generation(device_id, bandwidth, i, model_type)
#             for j in range(len(device_id)):
#                 pre_layer_selection[device_id[j]] = layer_selection[j]
  #  print(np.max(pre_delay))
    return pre_edge_assign, pre_layer_selection, np.max(pre_delay)
            

def hierfavg_algorithm(bandwidth, device_num, edge_num,model_length,str):
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]

    max_delay = 0
    edge_assign = np.zeros(device_num, dtype=np.int)  
    layer_selection = np.ones((device_num,model_length),dtype = np.int)
    for i in range(device_num):
        edge_assign[i] = i%edge_num
        delay = sum(model_size)/bandwidth[i][edge_assign[i]]
        if max_delay<delay:
            max_delay = delay
    return edge_assign, layer_selection, max_delay


def hfl_noniid(bandwidth,device_label_distribution,device_num, edge_num,data_type, model_length,str):
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]

    max_delay = 0
    if data_type == 'cifar10':
        label_num = 10
    elif data_type == "cifar100":
        label_num = 100
    elif data_type == 'image':
        label_num = 10
    elif data_type == 'emnist':
        label_num = 62
    #priority
    edge_assignment = np.zeros(device_num, dtype=np.int)
    for i in range(len(edge_assignment)):
        edge_assignment[i]=-1
    avaliable_device = np.ones(device_num, dtype=np.int)
    edk_global = np.zeros(label_num, dtype=np.float)
    edk_edge = np.zeros((edge_num,label_num),dtype = np.float)
    for i in range(len(edk_global)):
        edk_global[i] = float(1.0)/label_num

    maxiteration = 10
    while 1 in avaliable_device and maxiteration > 0:     
        maxiteration -= 1
        for i in range(edge_num):
            v = i
            edk_v_pre = copy.deepcopy(edk_edge[v])
            edk_v_after = copy.deepcopy(edk_edge[v])
            edk_v_pre_value = np.linalg.norm(edk_v_pre-edk_global,ord=1)
            for j in range(device_num):
                if avaliable_device[j] == 1:
                    u = j
                    for k in range(len(edk_v_pre)):
                        edk_v_after[k] = (edk_v_pre[k]*np.sum(edge_assignment[:]==v)+device_label_distribution[u][k])/(1+np.sum(edge_assignment[:]==v))
                    edk_v_after_value = np.linalg.norm(edk_v_after-edk_global,ord=1)
                    if edk_v_after_value <= edk_v_pre_value:
                        edge_assignment[u] = v
                        edk_edge[v] = copy.deepcopy(edk_v_after)
                        avaliable_device[u] = 0
                        break
    for i in range(len(edge_assignment)):
        if avaliable_device[i] == 1:
            edge_assignment[i] = random.randint(0,edge_num-1)

    layer_selection = np.ones((device_num,model_length),dtype = np.int)
    for i in range(device_num):
        delay = sum(model_size)/bandwidth[i][edge_assignment[i]]
        if max_delay<delay:
            max_delay = delay
    return edge_assignment, layer_selection, max_delay


def hfel(bandwidth,model_length,str):
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]

    max_delay = 0
    bandwidth = np.array(bandwidth)
    edge_number = len(bandwidth[0])
    device_number = len(bandwidth)
    edge_assignment = np.zeros(device_number, dtype=np.int)*(-1)
    for i in range(device_number):
        edge_assignment[i] = np.argmax(bandwidth[i])
    flag = True
    while flag:
        flag = False
        for i in range(edge_number):
            if not i in edge_assignment:
                edge_assignment[np.argmax(bandwidth[:,i])] = i
        for i in range(edge_number):
            if not i in edge_assignment:
                flag  = True
    layer_selection = np.ones((device_number,model_length),dtype = np.int)
    for i in range(device_number):
        delay = sum(model_size)/bandwidth[i][edge_assignment[i]]
        if max_delay<delay:
            max_delay = delay
    return edge_assignment, layer_selection, max_delay  


def Heals_random(bandwidth,edge_num,device_num,model_length,str):
    if str == 'NIN':
        model_size = [1.07,0.28,0.288,0,18.76,2.01,2.017,0,27.025,4.52,4.525,0,108.06,32.06,3.12,0] #Mb
    elif str == 'VGG':
        model_size = [0.060,1.13,0,2.26,4.51,0,9.02,18.02,18.02,0,36.05,72.05,72.05,0,72.05,72.05,72.05,0,32.06,128.06,6.25]
    elif str == 'VGG9':
        model_size = [0.031,0.570,0,2.2636,4.513 ,0,9.025,18.025,0,64.01,8.01660,0.1575]
    elif str == 'AlexNet':
        model_size = [0.06,0,3.39,0,20.28,27.02,18.025,0,64.063,128.06,0.626]
    elif str == 'LeNet':
        model_size = [0.005,0,0.074,0,12.5323,16.016, 0.971]

    max_delay = 0
    edge_assignment = np.zeros(device_num, dtype=np.int)*(-1)
    layer_selection = np.zeros((device_num,model_length),dtype = np.int)
    a = []
    for i in range(device_num):
        a.append(i)
    while len(a) >=1 :
     #   print(a)
        for i in range(edge_num):
            if len(a) == 0:
                break
            else:
                b = random.randint(0,len(a)-1)
                edge_assignment[a[b]] = i
                a.remove(a[b])
    edge_assignment_matric = edge_assignment_array_to_matric(edge_assignment, edge_num)
    for i in range(edge_num):
        for j in range(model_length):
            c = random.randint(1, len(edge_assignment_matric[i]))
            se = copy.deepcopy(edge_assignment_matric[i])
            for k in range(c):
                d = random.randint(0,len(se)-1)
                layer_selection[se[d]][j] = 1
                se.remove(se[d])
    for i in range(device_num):
        size = 0
        for j in range(len(model_size)):
            size += model_size[j]*layer_selection[i][j]
        delay = size/bandwidth[i][edge_assignment[i]]
        if max_delay<delay:
            max_delay = delay
    return edge_assignment, layer_selection, max_delay


# neibo_id = [0,1,2,3]
# node_num=0
# bandwidth = [[1],[2],[3],[4]]

# str = 'NIN'


# model_rec2 =  layer_selection_generation(neibo_id, bandwidth, node_num, str)
# print(model_rec2)


# label= [[0.2,0,0.2,0,0.2,0,0.2,0,0.2,0],
#         [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
#         [0,0.2,0,0.2,0,0.2,0,0.2,0,0.2]]

# layer = np.ones((3,12),dtype=int)

# bandwidth = [[1,1],
#                [1,2],
#                [1,2]]

# str = 'NIN'
# data = 'cifar10'

# result = heals_algorithm(label, bandwidth,str,data,12)

# print(result)


# bandwidth = [[1,1,3,4],
#                [1,2,0.2,0.5],
#                [1,2,8,2],
#                [1,5,3,4],
#                [1,2,0.2,0.5],
#                [3,8,1,2]]

# result = hfel(bandwidth,12)
# print(result)



