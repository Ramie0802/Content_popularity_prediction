
#----------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

class DataSet:

    def __init__(self):
        pass

    @classmethod
    def LoadDataSet(cls, name='TMDB'):
        
        
        ratings = pd.read_csv(f'./data/{name}.csv')
        ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        movie_dataset = pd.read_csv("./data/TMDB_movie_dataset_v11.csv")
        movie_dataset = movie_dataset[["id","popularity","genres"]]
        movie_dataset.columns = ["movie_id","popularity","genres"]


        dataset =  pd.merge(ratings, movie_dataset, on='movie_id', how='inner')
        
        # merge movie info
        dataset = dataset.dropna(subset = ['genres'])
        dataset = dataset.reset_index()

        dff = dataset.copy()
        genre_l = dff['genres'].apply(lambda x: x.split(','))
        genre_l = pd.DataFrame(genre_l)
        genre_l['genres'] = genre_l['genres'].apply(lambda x :[ y.strip().lower().replace(' ','') for y in x] )

        


        MLB = MultiLabelBinarizer()
        genre_encoded = MLB.fit_transform(genre_l['genres'])
        genre_encoded_df = pd.DataFrame(genre_encoded, columns=MLB.classes_)
        genre_encoded_df=genre_encoded_df.reset_index()
        mod_df = dff.drop(['genres'],axis=1).drop('index',axis=1)
        mod_df=mod_df.reset_index()

        dataset = pd.concat([mod_df,genre_encoded_df],axis=1).drop('index',axis=1)
        dataset = dataset.sort_values(by=['movie_id','timestamp'], ascending=True).reset_index()
        dataset['genres'] = genre_l['genres'] 
        dataset = dataset.drop('index', axis=1)
        


        columns_remove = ['rating', 'timestamp', 'popularity']    
        data = dataset.drop(columns_remove, axis=1)
        data['popularity'] = dataset['popularity']
        
        
        if name != '':            
            ratings = data.sort_values(by='user_id', ignore_index=True)
        print("Load " + name + " data_set success.\n")
        return ratings



