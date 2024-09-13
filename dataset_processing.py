import numpy as np
import pandas as pd 
import copy
import torch


from data_set import DataSet
from user_info import UserInfo
import  utils  





def get_dataset(args):
    """
    :param: args:
    :return: ratings: dataFrame ['user_id' 'movie_id' 'rating']
    :return: user_info:  dataFrame ['user_id' 'gender' 'age' 'occupation']
    """
    model_manager = utils.ModelManager('data_set')
    user_manager = utils.UserInfoManager(args.dataset)

    '''Do you want to clean workspace and retrain model/data_set user again?'''
    '''if you want to retrain model/data_set user, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_dataset)
    user_manager.clean_workspace(args.clean_user)

    # 导入模型信息
    try:
        ratings = model_manager.load_model(args.dataset + '-ratings')
        print("Load " + args.dataset + " data_set success.\n")
    except OSError:
        ratings = DataSet.LoadDataSet(name=args.dataset)
        model_manager.save_model(ratings, args.dataset + '-ratings')

    # 导入用户信息
    try:
        user_info = user_manager.load_user_info('user_info')
        print("Load " + args.dataset + " user_info success.\n")
    except OSError:
        user_info = UserInfo.load_user_info(name=args.dataset)
        user_manager.save_user_info(user_info, 'user_info')

    return ratings, user_info

def sampling_mobility(args,vehicle_num):
    """
    :param args
    :return: sample: matrix user_id|movie_id|rating|gender|age|occupation|label
    :return: user_group_train, the idx of sample for each client for training
    :return: user_group_test, the idx of sample for each client for testing
    """
    # 存储每个client信息
    model_manager = utils.ModelManager('clients')
    '''Do you want to clean workspace and retrain model/clients again?'''
    '''if you want to change test_size or retrain model/clients, please set clean_workspace True'''
    model_manager.clean_workspace(args.clean_clients)
    # 导入模型信息
    try:
        users_group_train = model_manager.load_model(args.dataset + '-user_group_train')
        users_group_test = model_manager.load_model(args.dataset + '-user_group_test')
        sample = model_manager.load_model(args.dataset + '-sample')
        print("Load " + args.dataset + " clients info success.\n")
    except OSError:
        # 调用get_dataset函数，得到ratings,user_info
        ratings, user_info = get_dataset(args)


        #ce_round
        
        users_num_client = np.random.randint(10, 15, 15) # fix số lượng xe ở đây
        print(users_num_client)
        users_num_client = sorted(users_num_client)

        a=0
        for i in range(15):
            a+=users_num_client[i]

        for i in range(vehicle_num):
            users_num_client[i] = int((user_info.index[-1] + 1) * users_num_client[i] / a )
        print('each vehicle allocated data:',users_num_client)

        user_seq_client=[]
        for i in range(vehicle_num):
            num=0
            for j in range(i):
                num+=users_num_client[j]
            user_seq_client.append(num)

        
        sample = pd.merge(ratings, user_info, on=['user_id'], how='inner')
        #sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64', 'timestamp':'int64'})
        
        


        users_group_all, users_group_train, users_group_test ,request_content= {}, {}, {}, {}
       
        all_test_num = 0
        for i in range(vehicle_num):
            print('loading client ' + str(i))
            index_begin = ratings[ratings['user_id'] == user_seq_client[i] + 1].index[0]         
            index_end = ratings[ratings['user_id'] == user_seq_client[i] + users_num_client[i] ].index[-1]
            users_group_all[i] = set(np.arange(index_begin, index_end + 1))

            NUM_train = int(0.998 * len(users_group_all[i]))

            users_group_train[i] = set(np.random.choice(list(users_group_all[i]), NUM_train, replace=False))
            users_group_test[i] = users_group_all[i] - users_group_train[i]
            # 将set转换回list，并排序
            users_group_train[i] = list(users_group_train[i])
            users_group_test[i] = list(users_group_test[i])
            users_group_train[i].sort()
            users_group_test[i].sort()

            all_test_num += NUM_train / 0.998 * 0.002

            print('generate client ' + str(i) + ' info success\n')
        print('all_test_num',all_test_num)
        vehicle_request_num = dict([(k, []) for k in range(args.epochs)])

        for i in range(args.epochs):
            for j in range(vehicle_num):
                if j==0:
                    vehicle_request_num[i]=[]
                    request_content[i]=np.random.choice(list(users_group_test[j]), int(all_test_num / args.epochs * users_num_client[j] / a), replace=True)
                    vehicle_request_num[i].append(int(all_test_num / args.epochs*users_num_client[j]/a))
                else:
                    request_content[i]=np.append(request_content[i],np.random.choice(list(users_group_test[j]), int(all_test_num / args.epochs *users_num_client[j]/a), replace=True))
                    vehicle_request_num[i].append(int(all_test_num / args.epochs * users_num_client[j] / a))

            request_content[i]=list(set(request_content[i]))
            request_content[i].sort()


        # 存储user_group_train user_group_test sample
        model_manager.save_model(users_group_train, args.dataset + '-user_group_train')
        model_manager.save_model(users_group_test, args.dataset + '-user_group_test')

    return sample, users_group_train, users_group_test, request_content, vehicle_request_num



def preprocessing_federated_branch(sample):

    columns_feature = ['user_id', 'movie_id', 'action', 'adventure', 'animation',
       'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy',
       'history', 'horror', 'music', 'mystery', 'romance', 'sciencefiction',
       'thriller', 'tvmovie', 'war', 'western', 'popularity']
    dataset = sample[columns_feature]
    return dataset

def preprocessing_lstm_branch(sample):
    columns_feature = ['user_id', 'movie_id', 'action', 'adventure', 'animation',
       'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy',
       'history', 'horror', 'music', 'mystery', 'romance', 'sciencefiction',
       'thriller', 'tvmovie', 'war', 'western', 'popularity']
    dataset = sample[columns_feature]
    return dataset



def preprocessing_lstm_semantic_branch(sample):
    columns_feature = ['user_id', 'movie_id','genres', 'popularity']
    dataset = sample[columns_feature]

    custom_embedding_path = 'custom_embedding.txt'
    custom_embedding = load_word_embedding(custom_embedding_path)
    # Map genre to vector
    genre_to_vector = {genre: vector for genre, vector in custom_embedding.items()}
    # Create vectors for each ID movie based on genres
    id_vectors = {}
    for id_val, genres in zip(dataset['movie_id'], dataset['genres']):
        id_vector = []
        for genre in genres:
            genre_vector = genre_to_vector.get(genre)
            if genre_vector:
                id_vector.append(genre_vector)
                sum_list = [sum(sublist) for sublist in zip(*id_vector)]
        id_vectors[id_val] = sum_list

    
    id_to_vector = {str(key): id_vectors.get(key) for key in dataset['movie_id']}
    dataset['semantic_vector'] = [id_to_vector.get(str(id_val)) for id_val in dataset['movie_id']]
    dataset = pd.concat([dataset.drop('semantic_vector', axis=1), dataset['semantic_vector'].apply(pd.Series)], axis=1)

    columns_remove = ['user_id','genres', 'popularity']    

    X = dataset.drop(columns_remove, axis=1)
    y = dataset['popularity']

    final_dataset = X
    final_dataset['popularity'] = y
   

    return final_dataset

def preprocessing_cnn_branch(sample):
    columns_feature = ['user_id', 'movie_id', 'action', 'adventure', 'animation',
       'comedy', 'crime', 'documentary', 'drama', 'family', 'fantasy',
       'history', 'horror', 'music', 'mystery', 'romance', 'sciencefiction',
       'thriller', 'tvmovie', 'war', 'western', 'popularity']
    dataset = sample[columns_feature]
    return dataset



def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg



def load_word_embedding(file_path):
    embedding_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            embedding_dict[word] = vector
    return embedding_dict