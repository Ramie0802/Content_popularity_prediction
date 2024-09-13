
import torch
import numpy as np

import pandas as pd
from dataset_processing import preprocessing_lstm_semantic_branch

def test(model, sample, users_idx_test):
    rows = sample.loc[users_idx_test]
    
    df_test = rows.drop('popularity', axis=1)

    x_test = torch.tensor(df_test.values, dtype=torch.float32) 
    
    with torch.no_grad():
        predictions = model(x_test) 
    
    predict_list = [tensor.item() for tensor in predictions]

    df_test['popularity'] =  predict_list
    finallist = df_test

    return finallist


def test_lstm(model, sample, users_idx_test):

    rows = sample.loc[users_idx_test] 
    data_set_test = np.array(rows)  

    data_set_test = torch.tensor(data_set_test, dtype=torch.float32) 
    x_test = data_set_test[:,:-1]
    
    with torch.no_grad():
        predictions = model(x_test.unsqueeze(1)) 
    
    predict_list = [tensor.item() for tensor in predictions]

    df_test = rows
    df_test['popularity'] = predict_list

    finallist = df_test

    return finallist


def test_cnn(model, sample, users_idx_test):

    rows = sample.loc[users_idx_test] 
    data_set_test = np.array(rows)  

    data_set_test = torch.tensor(data_set_test, dtype=torch.float32) 
    x_test = data_set_test[:,:-1]
    
    with torch.no_grad():
        predictions = model(x_test) 
    
    predict_list = [tensor.item() for tensor in predictions]

    df_test = rows
    df_test['popularity'] = predict_list

    finallist = df_test

    return finallist



