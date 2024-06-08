import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL


def get_similar(embedding, k):
    model_similar_items = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(embedding)
    distances, indices = model_similar_items.kneighbors(embedding)
    
    return distances, indices

def show_similar(item_name, item_similar_indices, item_encoder, df_items):
    
    item_name_lower = item_name.lower()

    
    # Get the item index from the item name
    item_index = df_items[df_items['cleaned_title'].str.lower() == item_name_lower]['movie_id'].values[0]
    # Encode the item index using the item encoder
    item_encoded = item_encoder.transform([item_index])[0]
        
    s = item_similar_indices[item_index]
    movie_ids = item_encoder.inverse_transform(s)

    images = []
    for movie_id in movie_ids:
        img_path = 'data/posters/' + str(movie_id) + '.jpg'
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.axis('off')
        plt.imshow(image)