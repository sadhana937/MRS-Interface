import sys
import warnings
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Add, Activation
from keras.regularizers import l2

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
df_ratings = pd.read_csv("data/ratings.csv")
df_items = pd.read_csv("data/items.csv")

# Define functions (you may need to replace this with actual preprocessing functions)
def encode_user_item_withencoder(df, user_col, item_col, rating_col, timestamp_col):
    user_ids = df[user_col].astype('category').cat.codes
    item_ids = df[item_col].astype('category').cat.codes
    df['USER'] = user_ids
    df['ITEM'] = item_ids
    df['RATING'] = df[rating_col]
    return df, user_ids, item_ids

def random_split(df, split_ratio):
    train = df.sample(frac=split_ratio[0], random_state=42)
    test = df.drop(train.index)
    return train, test

# Apply preprocessing
DATA, user_encoder, item_encoder = encode_user_item_withencoder(df_ratings, "user_id", "movie_id", "rating", "unix_timestamp")

# Define model parameters
n_users = DATA.USER.nunique()
n_items = DATA.ITEM.nunique()
n_factors = 50

# Define the model
def ExplicitMF(n_users, n_items, n_factors):
    item_input = Input(shape=[1], name='Item')
    item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6), name='ItemEmbedding')(item_input)
    item_vec = Flatten(name='FlattenItemsE')(item_embedding)

    user_input = Input(shape=[1], name='User')
    user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6), name='UserEmbedding')(user_input)
    user_vec = Flatten(name='FlattenUsersE')(user_embedding)

    prod = Dot(name='DotProduct', axes=1)([item_vec, user_vec])
    
    user_bias = Embedding(n_users, 1, name='UserBias')(user_input)
    item_bias = Embedding(n_items, 1, name='ItemBias')(item_input)
    
    output = Add(name='AddBias')([prod, user_bias, item_bias])
    output = Flatten()(output)
    output = Activation('sigmoid')(output)
    
    model = Model([user_input, item_input], output)
    return model

# Create and compile the model
model = ExplicitMF(n_users, n_items, n_factors)
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data
train, test = random_split(DATA, [0.8, 0.2])

# Train the model
history = model.fit([train.USER, train.ITEM], train.RATING, epochs=10, verbose=1, validation_split=0.1)

# Save the model in H5 format
model.save('explicit_mf_model.h5')
