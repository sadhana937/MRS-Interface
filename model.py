import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from IPython.display import SVG, display
#import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError  # Correct import
from reco.preprocess import encode_user_item_withencoder, random_split, user_split, remove_year

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Lambda, Activation, Reshape
from keras.regularizers import l2
from keras.constraints import non_neg
from keras.utils import plot_model
from keras.utils import model_to_dot
from reco.evaluate import get_embedding, get_predictions, recommend_topk


df_ratings = pd.read_csv("data/ratings.csv")
df_items = pd.read_csv("data/items.csv")

DATA, user_encoder, item_encoder = encode_user_item_withencoder(df_ratings, "user_id", "movie_id", "rating", "unix_timestamp")
df_items['cleaned_title'] = df_items['title'].apply(remove_year)

n_users = DATA.USER.nunique()
n_items = DATA.ITEM.nunique()

max_rating = DATA.RATING.max()
min_rating = DATA.RATING.min()

train, test = random_split(DATA, [0.8, 0.2])

def NNMF_Bias(n_users, n_items, n_factors):
    
    # Item Layer
    item_input = Input(shape=[1], name='Item')
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-5),
                               embeddings_constraint= non_neg(),
                               name='ItemEmbedding')(item_input)
    item_vec = Flatten(name='FlattenItemE')(item_embedding)
    
    # Item Bias
    item_bias = Embedding(n_items, 1, embeddings_regularizer=l2(1e-5), name='ItemBias')(item_input)
    item_bias_vec = Flatten(name='FlattenItemBiasE')(item_bias)

    # User Layer
    user_input = Input(shape=[1], name='User')
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-5), 
                               embeddings_constraint= non_neg(),
                               name='UserEmbedding')(user_input)
    user_vec = Flatten(name='FlattenUserE')(user_embedding)
    
    # User Bias
    user_bias = Embedding(n_users, 1, embeddings_regularizer=l2(1e-5), name='UserBias')(user_input)
    user_bias_vec = Flatten(name='FlattenUserBiasE')(user_bias)

    # Dot Product of Item and User & then Add Bias
    DotProduct = Dot(axes=1, name='DotProduct')([item_vec, user_vec])
    AddBias = Add(name="AddBias")([DotProduct, item_bias_vec, user_bias_vec])
    
    # Scaling for each user
    rating_output = Activation('sigmoid')(AddBias)
    
    # Model Creation
    model = Model([user_input, item_input], rating_output)
    
    # Custom loss function to scale the output
    def custom_loss(y_true, y_pred):
        y_pred_scaled = y_pred * (max_rating - min_rating) + min_rating
        return MeanSquaredError()(y_true, y_pred_scaled) 
    
    # Compile Model
    model.compile(loss=custom_loss, optimizer="adam")
    
    return model

def custom_loss(y_true, y_pred):
        y_pred_scaled = y_pred * (max_rating - min_rating) + min_rating
        return MeanSquaredError()(y_true, y_pred_scaled) 

n_factors = 40
model = NNMF_Bias(n_users, n_items, n_factors)

output = model.fit([train.USER, train.ITEM], train.RATING, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

model.save('nnmf_model.h5')

# import pickle

# # Save the model
# with open('nnmf_model.pkl', 'wb') as f:
#     pickle.dump(model, f)



# item_embedding = get_embedding(model, "ItemEmbedding")
# user_embedding = get_embedding(model, "UserEmbedding")

# from reco.recommend import get_similar, show_similar

# item_distances, item_similar_indices = get_similar(item_embedding, 5)

# show_similar("Apollo 13", item_similar_indices, item_encoder, df_items)



