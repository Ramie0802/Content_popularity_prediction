import sys
import copy
import time
import numpy as np
import pandas as pd
from itertools import chain

import torch

from utils import  get_top_items

from options import args_parser
from dataset_processing import sampling_mobility,  preprocessing_cnn_branch 

from predict import test_cnn

from model import  CNNModel
from local_update import LocalUpdate_branch3




if __name__=='__main__':
      
    
    args = args_parser()
    idx = 0
    
    if args.gpu: torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    sample, users_group_train, users_group_test, request_content, vehicle_request_num = sampling_mobility(args,
                                                                             args.clients_num)
    
    # dataset
    
    sample_cnn_branch = preprocessing_cnn_branch(sample)
    print(sample_cnn_branch)

    data_set_cnnbranch = np.array(sample_cnn_branch) 

     
           
    
         
    print('different epoch vehicle request num', vehicle_request_num)
    request_dataset = []
    for i in range(args.epochs):
        request_dataset_idxs = []
        print(request_content[i])
        request_dataset_idxs.append(request_content[i])
        request_dataset_idxs = list(chain.from_iterable(request_dataset_idxs))
        request_dataset = request_dataset_idxs
        
      
           
    
    
    dim_in3 = data_set_cnnbranch.shape[1]-1

    # federated learning branch
    initial_cnnmodel = model = CNNModel()
    initial_cnnmodel.to(device)
    initial_cnnmodel.train()


    # khoi tao mo hinh dau tien cho tung vehicle - cac mo hinh update se duoc append
    vehicle_model_dict3 = [[] for _ in range(args.clients_num)]
    for i in range(args.clients_num):
        vehicle_model_dict3[i].append(initial_cnnmodel)


    # each epoch train time
    each_epoch_time=[]
    each_epoch_time.append(0)
   


    while idx < args.epochs:        
                    
        epoch_start_time = time.time()

        local_net3 = copy.deepcopy(vehicle_model_dict3[idx % args.clients_num][-1])
        local_net3.to(device)     
        
        for veh in range(args.clients_num):
            local_model3 = LocalUpdate_branch3(args=args, dataset=data_set_cnnbranch
    ,
                                        idxs=users_group_train[veh % args.clients_num])
            w3, loss3, local_net3 = local_model3.update_weights(
                model=local_net3, client_idx=veh % args.clients_num + 1, global_round=idx + 1)
            vehicle_model_dict3[veh].append(copy.deepcopy(local_net3))# cho nay phai copy weight cuar cluster
    
        recommend_movies_c500 = []
        for veh in range(args.clients_num):
            vehicle_seq = veh
            recommend_list = test_cnn(vehicle_model_dict3[veh][-1], sample_cnn_branch, users_group_test[vehicle_seq])
            recommend_list500 = get_top_items(recommend_list, 0.25)
            recommend_movies_c500.append(recommend_list500)
            recommend_movies_c5003_df = pd.concat(recommend_movies_c500, axis=0)
        print(f"List recommed movies at BS at round {idx + 1}", recommend_movies_c5003_df)
        
        epoch_time = time.time() - epoch_start_time
        each_epoch_time.append(epoch_time)
        
        idx += 1                  

        if idx > args.epochs:
            break


        
        


