#Importing the necessary libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import StandardScaler
from surprise.model_selection import cross_validate,GridSearchCV
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline,SVD
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import f_oneway
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
from surprise import SVD,Reader, Dataset
from surprise.prediction_algorithms.knns import KNNBasic
import warnings
warnings.filterwarnings("ignore")
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')



class load_data:
    def __init__(self,filepath):
        self.filepath=filepath
        self.df=pd.read_csv(self.filepath)
        print("Data loaded successfully.")
              
    def get_data(self):
        return self.df
    

    
def explore_data(df):
    """
    - Shape of the dataset
    - Info (data types, non-null counts)
    - Statistical summary (describe)
    
    """
    # Displaying the first 5 rows
    print("displaying the first 5 rows","\n", df.head(),"\n")
    
    # Display shape
    print("Shape of the dataset: ", df.shape, "\n")
        
    # Display info
    print("Dataset info:\n")
    df.info()
    print("\n")
        
    # Display statistical summary
    print("Statistical summary:\n", df.describe(), "\n")
    
    #Checking for nulls
    nan_percent = (df.isna().sum() / len(df))*100 # total percent of missing values per column
    print("percentage of nulls","\n", nan_percent)
    


def visualize_outliers_with_boxplot(data, column_name):
    """
    Creates a boxplot to visualize potential outliers in a specific column.

    Parameters:
    data (pd.df): The dataframe containing the data.
    column_name (str): The column name to visualize outliers for.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.boxplot(x=data[column_name], color='skyblue')  
    plt.title(f'Boxplot of {column_name}')  
    plt.xlabel(column_name)  
    plt.show()  
    
    
def check_and_remove_duplicates(df):
    """
    Checks for duplicate rows in a DataFrame and removes them if any exist.
    
    Parameters:
    df (pd.df): The DataFrame to check for duplicates.
    
    Returns:
    pd.df: The cleaned DataFrame without duplicates.
    """
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    
    if duplicate_count > 0:
        print("Removing duplicates...")
        df = df.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicates found.")
    
    return df



def create_sparsitymatrix(df):
    #Identify the unique users and movies in our data
    num_users = df["userId"].nunique()
    num_movies = df["movieId"].nunique()
    
    #Create mappings for user ids and movie ids 
    user_mapper = {user_id: i for i, user_id in enumerate(df["userId"].unique())}
    movie_mapper = {movie_id: i for i, movie_id in enumerate(df["movieId"].unique())}
    
    #Inverse mappings for users and movies IDs

    user_inv_mapper = {v: k for k, v in user_mapper.items()}
    movie_inv_mapper = {v: k for k, v in movie_mapper.items()}
    
    #Map users and Movie IDs to indices

    user_idx = df["userId"].map(user_mapper)
    movie_idx = df["movieId"].map(movie_mapper)
    
    #Creates sparse matrix
    X = csr_matrix((df["rating"].values, (movie_idx, user_idx)), shape=(num_movies, num_users))
    
    #Calculating sparsity for our data

    sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper, sparsity




current_user_id = 900  # Initialize global user ID

def movie_rater(movie_df):
    global current_user_id  # Access the global variable
    rating_list = []

    # Prompt the user to input the number of movies to rate
    num = int(input("Enter the number of movies you want to rate:\n").strip())

    # Prompt the user to input a genre 
    genre = input("Enter a genre to filter movies (or press Enter to skip):\n").strip()
    genre = genre if genre else None  # Set to None if the input is empty

    movies_rated = 0  # Counter for the number of movies successfully rated

    while movies_rated < num:
        # Select a random movie, filtered by genre if specified
        # lowercase genres in the DataFrame for case-insensitive matching
        movie = movie_df[movie_df['genres'].str.lower().str.contains(genre, na=False)].sample(1) if genre else movie_df.sample(1)
        print(movie)

        # Prompt the user to rate the movie
        rating = input('Rate this movie (1-5) or press "n" if you have not seen it:\n').strip().lower()
        print(f"You rated this movie: {rating}")
        if rating == 'n':
            print("Not Rated")
            continue  # Skip to the next movie if the user hasn't seen this one

        # Add the user's rating to the list
        rating_list.append({
            'userId': current_user_id,
            'movieId': movie['movieId'].values[0],
            'rating': float(rating)
        })
        movies_rated += 1  # Increment the count of successfully rated movies

    current_user_id += 1  # Increment the user ID for the next user
    return rating_list




def recommend_movies(user_ratings, movie_df, svd, genre=None, num_recommendations=5):
    """
    Recommends movies for a user based on the movies they have rated.

    Parameters:
    - user_ratings: List of dictionaries with keys 'userId', 'movieId', and 'rating'.
    - movie_df: DataFrame containing movie details with columns 'movieId' and 'title'.
    - svd: Trained SVD model for making predictions.
    - genre: (Optional) Genre to filter the movie recommendations by.
    - num_recommendations: Number of movies to recommend (default is 5).

    Returns:
    - List of recommended movies (titles).
    """
    # Get the user ID from the ratings
    user_id = user_ratings[0]['userId'] if user_ratings else None
    if not user_id:
        print("No user ratings provided.")
        return []

    # Get the list of movies the user has already rated
    rated_movie_ids = [rating['movieId'] for rating in user_ratings]
    
    # Convert the new ratings to a DataFrame
    new_ratings_df = pd.DataFrame(user_ratings)

    # Combine the new ratings with the existing dataset (assuming 'existing_ratings' is the original dataset)
    updated_ratings = pd.concat([rating_df, new_ratings_df])

    # Prepare the data for Surprise
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(updated_ratings[['userId', 'movieId', 'rating']], reader)

    # refitting the model on the new dataframe
    trainset_2 = surprise_data.build_full_trainset()
    svd.fit(trainset_2)
    
    # Find movies the user hasn't rated and drop duplicates due to the exploded "genres" column
    unrated_movies = movie_df[~movie_df['movieId'].isin(rated_movie_ids)].drop_duplicates(subset='movieId')
    
    # If genre filter is provided, apply it
    if genre:
        genre = genre.lower()
        # Ensure the genre is part of any of the movie's genres (case-insensitive)
        unrated_movies = unrated_movies[unrated_movies['genres'].str.contains(genre, na=False, case=False)]

    # Predict ratings for unrated movies
    unrated_movies['predicted_rating'] = unrated_movies['movieId'].apply(
        lambda movie_id: svd.predict(user_id, movie_id).est
    )

    # Sort movies by predicted rating in descending order
    recommendations = (
        unrated_movies.sort_values(by='predicted_rating', ascending=False)
        .head(num_recommendations)
    )
    
    # Explanation message
    print(
        f"Based on the {len(user_ratings)} movies you've rated, "
        "here are some recommendations tailored to your preferences. "
        "These movies have high predicted ratings, suggesting you might enjoy them!"
    )
    
    # Return the recommended movie titles
    return recommendations[['movieId','title','predicted_rating']]



def get_hybrid_recommendations(user_ratings, movie_df, svd, num_recommendations=5, alpha=0.7):
    """
    Generate hybrid recommendations using collaborative filtering and content-based filtering.

    Parameters:
    - user_ratings: List of dictionaries with keys 'userId', 'movieId', and 'rating'.
    - movie_df: DataFrame containing movie details with columns 'movieId', 'title', and 'genres'.
    - svd: Trained SVD model for collaborative filtering.
     num_recommendations: Number of movies to recommend (default is 5).
    - alpha: Weight for collaborative filtering in the hybrid recommendation (default is 0.7).

    Returns:
    - DataFrame with hybrid recommended movies and their details.
    """
    # Collaborative filtering scores
    recommendations = recommend_movies(user_ratings, movie_df, svd, num_recommendations=None)
    
    # Ensure movieId column exists in recommendations
    collaborative_scores = recommendations[['movieId', 'predicted_rating']].drop_duplicates('movieId')
    
    # Merge collaborative scores into the movie_df
    movie_df = movie_df.merge(collaborative_scores, on='movieId', how='left', suffixes=('', '_collab'))
    movie_df['collaborative_scores'] = movie_df['predicted_rating'].fillna(0)
    
    
   # Content-based filtering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import MultiLabelBinarizer

    # Prepare genre encoding
    mlb = MultiLabelBinarizer()
    movie_df['genres'] = movie_df['genres'].fillna('')  # Handle missing genres
    genre_encoded = mlb.fit_transform(movie_df['genres'])
    genre_similarity = cosine_similarity(genre_encoded)
    
    # Calculate content-based scores
    content_based_scores = genre_similarity.mean(axis=0)
    movie_df['content_based_scores'] = content_based_scores

    # Hybrid scores
    movie_df['hybrid_scores'] = alpha * movie_df['collaborative_scores'] + (1 - alpha) * movie_df['content_based_scores']
    
    # Get top recommendations based on hybrid scores
    top_recommendations = movie_df.sort_values(by='hybrid_scores', ascending=False).drop_duplicates('movieId').head(num_recommendations)

    return top_recommendations[['title', 'genres', 'hybrid_scores']]









   


      

            
            






