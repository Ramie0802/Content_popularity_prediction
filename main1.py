import sys
import copy
import time
import numpy as np
import pandas as pd
from itertools import chain

import torch

from utils import  get_top_items

from options import args_parser
from dataset_processing import sampling_mobility, average_weights, preprocessing_federated_branch, preprocessing_lstm_semantic_branch

from predict import test
from cluster import clustering

from model import FedModel
from local_update import LocalUpdate




if __name__=='__main__':
      
    
    args = args_parser()
    idx = 0
    
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args,
                                                                             args.clients_num)
    
    # dataset
    dataset_federated_branch = preprocessing_federated_branch(sample)
    

    data_set_fedbranch = np.array(dataset_federated_branch)  
           
    
         
    print('different epoch vehicle request num', vehicle_request_num)
    request_dataset = []
    for i in range(args.epochs):
        request_dataset_idxs = []
        print(request_content[i])
        request_dataset_idxs.append(request_content[i])
        request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs))
        request_dataset = request_dataset_idxs
        print("\nrequest data set: ", sample.loc[request_dataset_idxs])
        
           
    
    dim_in1 = data_set_fedbranch.shape[1]-1
 

    # federated learning branch
    global_model = FedModel(dim_in1) 

    global_model.to(device)
    global_model.train()


    
    vehicle_model_dict = [[] for _ in range(args.clients_num)]
    for i in range(args.clients_num):
        vehicle_model_dict[i].append(copy.deepcopy(global_model))

    w_all_epochs = dict([(k, []) for k in range(args.epochs)])

    train_loss = []

    # each epoch train time
    each_epoch_time=[]
    each_epoch_time.append(0)


    cache_efficiency_list=[]
    cache_efficiency_without_list=[]

    request_delay_list=[]
   


    while idx < args.epochs:        
                    
        epoch_start_time = time.time()

        local_net = copy.deepcopy(vehicle_model_dict[idx % args.clients_num][-1]) # lay model moi nhat trong list model da train
        local_net.to(device)

        local_weights_avg=[]            
        
        for veh in range(args.clients_num):
            local_model = LocalUpdate(args=args, dataset=data_set_fedbranch
    ,
                                        idxs=users_group_train[veh % args.clients_num])
            w, loss, local_net = local_model.update_weights(
                model=local_net, client_idx=veh % args.clients_num + 1, global_round=idx + 1)
            local_weights_avg.append(copy.deepcopy(w))
            vehicle_model_dict[veh].append(copy.deepcopy(local_net))# cho nay phai copy weight cuar cluster

        
        # cluster ở đây
        weights = []
        for mod in local_weights_avg:
            all_weight = torch.cat([param.flatten() for param in mod.values()])
            weights.append(all_weight.tolist())
       
        
        
        vehicle_features_list = np.array(weights)
        clusters_veh, best_n_clusters = clustering(args, vehicle_features_list)       

        exit()
    
        local_weights_avg_lst = [[] for _ in range(best_n_clusters)]
        global_weights_avg_lst = [[] for _ in range(best_n_clusters)]
        global_model_lst = []

        for cluster in range(best_n_clusters):
            global_model_lst.append(FedModel(dim_in1))
            local_weights_avg_lst[cluster] = []
            for veh in clusters_veh[cluster]:
                local_weights_avg_lst[cluster].append(copy.deepcopy(w))

        global_weights_avg_lst[cluster] = average_weights(local_weights_avg_lst[cluster])
        global_model_lst[cluster].load_state_dict(global_weights_avg_lst[cluster]) 


        
        epoch_time = time.time() - epoch_start_time
        each_epoch_time.append(epoch_time)

        # test
        cache_size=100       
        recommend_movies_c500_cluster = [[] for _ in range(best_n_clusters)]
        recommend_movies_c500 = []
        for cluster in range(best_n_clusters):
            for i in clusters_veh[cluster]:
                vehicle_seq = i
                recommend_list = test(global_model_lst[cluster], dataset_federated_branch, users_group_test[vehicle_seq])                
                recommend_list500 = get_top_items(recommend_list)
                recommend_movies_c500.extend(list(recommend_list500))
        
        print(f"List recommed movies at BS at round {idx}", recommend_movies_c500)



        
        idx += 1                  

        if idx > args.epochs:
            break


        
        


