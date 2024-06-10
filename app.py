from keras.models import load_model
from model import item_encoder

# Load the model
model = load_model('nnmf_model.h5', compile=False)

# Define or import the custom loss function and any other necessary functions
#from model import custom_loss

# Compile the model with the custom loss function
#model.compile(loss=custom_loss, optimizer="adam")


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from reco.preprocess import remove_year


# Load the necessary files and models
df_items = pd.read_csv('data/items.csv')
df_items['cleaned_title'] = df_items['title'].apply(remove_year)


# Initialize the Flask app
app = Flask(__name__)

# Function to get similar items
def get_similar(embedding, k):
    model_similar_items = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(embedding)
    distances, indices = model_similar_items.kneighbors(embedding)
    return distances, indices

# Function to show similar items and return poster URLs
def show_similar(item_name, item_similar_indices, item_encoder, df_items):
    item_name_lower = item_name.lower()
    
    # Get the item index from the item name
    item_index = df_items[df_items['cleaned_title'].str.lower().str.contains(item_name_lower)].index[0]
    
    # Get similar items
    similar_indices = item_similar_indices[item_index][1:]  # Exclude the input movie itself
    similar_movie_ids = item_encoder.inverse_transform(similar_indices)

    # Get the poster URLs for the recommended movies
    posters = []
    for movie_id in similar_movie_ids:
        img_path = f'static/posters/{movie_id}.jpg'
        posters.append(img_path)
    
    return posters

# Get item embeddings
item_embedding = model.get_layer('ItemEmbedding').get_weights()[0]

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form['movie'].strip().lower()
    
    # Get similar items
    _, item_similar_indices = get_similar(item_embedding, 6)  # Get top 5 similar items + the item itself

    # Get the posters for the recommended movies
    posters = show_similar(movie_name, item_similar_indices, item_encoder, df_items)

    return render_template('predict.html', posters=posters, movie_name = request.form['movie'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
