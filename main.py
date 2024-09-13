import sys
import copy
import time
import numpy as np
import pandas as pd
from itertools import chain

import torch

from utils import  get_top_items, concate3branch

from options import args_parser
from dataset_processing import sampling_mobility, average_weights, preprocessing_federated_branch, preprocessing_lstm_semantic_branch, preprocessing_cnn_branch

from predict import test, test_lstm, test_cnn
from cluster import clustering

from model import FedModel, LSTMModel, CNNModel
from local_update import LocalUpdate, LocalUpdate_branch2, LocalUpdate_branch3




if __name__=='__main__':
      
    
    args = args_parser()
    idx = 0
    
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args,
                                                                             args.clients_num)
    
    # dataset
    sample_federated_branch = preprocessing_federated_branch(sample)
    data_set_fedbranch = np.array(sample_federated_branch)   
    
    
    sample_lstm_branch = preprocessing_lstm_semantic_branch(sample)
    data_set_lstmbranch = np.array(sample_lstm_branch)    

    sample_cnn_branch = preprocessing_cnn_branch(sample)
    data_set_cnnbranch = np.array(sample_cnn_branch) 
    
         
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
    dim_in2 = data_set_lstmbranch.shape[1]-1
 

    # federated learning branch
    global_model = FedModel(dim_in1)   
    global_model.to(device)
    global_model.train()

    # download initial model for each vehicle
    vehicle_model_dict1 = [[] for _ in range(args.clients_num)]
    for i in range(args.clients_num):
        vehicle_model_dict1[i].append(copy.deepcopy(global_model))

    #lstm branch
    initial_model = LSTMModel(dim_in2,hidden_size=64, output_size=1 )
    initial_model.to(device)
    initial_model.train()   

    # initial model for each vehicle
    vehicle_model_dict2 = [[] for _ in range(args.clients_num)]
    for i in range(args.clients_num):
        vehicle_model_dict2[i].append(initial_model)
        

    #cnn branch
    initial_cnnmodel = model = CNNModel()
    initial_cnnmodel.to(device)
    initial_cnnmodel.train()

    # initial model for each vehicle
    vehicle_model_dict3 = [[] for _ in range(args.clients_num)]
    for i in range(args.clients_num):
        vehicle_model_dict3[i].append(initial_cnnmodel)
 
 
    while idx < args.epochs:                            
        
        # FEDERATED BRANCH
        local_net = copy.deepcopy(vehicle_model_dict1[idx % args.clients_num][-1]) # get the lasted model
        local_net.to(device)        

        local_weights_avg=[]            
        
        for veh in range(args.clients_num):
            local_model = LocalUpdate(args=args, dataset=data_set_fedbranch,
                                        idxs=users_group_train[veh % args.clients_num])
            w, loss, local_net = local_model.update_weights(
                model=local_net, client_idx=veh % args.clients_num + 1, global_round=idx + 1)
            local_weights_avg.append(copy.deepcopy(w))
            vehicle_model_dict1[veh].append(copy.deepcopy(local_net))  # 

        
        # cluster here
        weights = [model['linear5.weight'] for model in local_weights_avg] 
        flattened_weights = [w.flatten().tolist() for w in weights]
    
                
        vehicle_features_list = np.array(flattened_weights)
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
        
        

        #LSTM - SEMANTIC BRANCH
        local_net2 = copy.deepcopy(vehicle_model_dict2[idx % args.clients_num][-1])
        local_net2.to(device)  

        for veh in range(args.clients_num):
            local_model2 = LocalUpdate_branch2(args=args, dataset=data_set_lstmbranch,
                                        idxs=users_group_train[veh % args.clients_num])
            w2, loss2, local_net2 = local_model2.update_weights(
                model=local_net2, client_idx=veh % args.clients_num + 1, global_round=idx + 1)
            vehicle_model_dict2[veh].append(copy.deepcopy(local_net2))

        #CNN BRANCH
        local_net3 = copy.deepcopy(vehicle_model_dict3[idx % args.clients_num][-1])
        local_net3.to(device)
        for veh in range(args.clients_num):
            local_model3 = LocalUpdate_branch3(args=args, dataset=data_set_cnnbranch
    ,
                                        idxs=users_group_train[veh % args.clients_num])
            w3, loss3, local_net3 = local_model3.update_weights(
                model=local_net3, client_idx=veh % args.clients_num + 1, global_round=idx + 1)
            vehicle_model_dict3[veh].append(copy.deepcopy(local_net3))



        # test - predict popularity 3 branch
        cache_size=100 

        recommend_movies_c500 = []
        for cluster in range(best_n_clusters):
            for veh in clusters_veh[cluster]:
                
                recommend_df1 = test(global_model_lst[cluster], sample_federated_branch, users_group_test[veh])           
                recommend_df2 = test_lstm(vehicle_model_dict2[veh][-1], sample_lstm_branch, users_group_test[veh])     
                recommend_df3 = test_cnn(vehicle_model_dict3[veh][-1], sample_cnn_branch, users_group_test[veh])

                recommend_3brabch = concate3branch(recommend_df1,recommend_df2, recommend_df3)

                recommend_movies_c500.append(recommend_3brabch)
                recommend_movies_c5001_df = pd.concat(recommend_movies_c500, axis=0)
                
        print(f"--prediction_df--round{idx+1}\n: ", recommend_movies_c5001_df)
                
        
        

       
        idx += 1                  

        if idx > args.epochs:
            break


        
        


