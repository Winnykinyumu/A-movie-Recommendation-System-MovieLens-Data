<img src="\movie-rec.jpg" alt="box office Image" style="width:11500px;height:400px;">


## PROJECT TITLE:
***

A MOVIE RECOMMENDATION SYSTEM USING MOVIELENS DATA

# Project overview
MovieLens Recommendation System
This project leverages the MovieLens dataset to build a robust movie recommendation system. It combines collaborative filtering, content-based filtering, and a hybrid recommendation approach to suggest personalized movie recommendations to users based on their past ratings and preferences.

# Project Description
The MovieLens Recommendation System is a data-driven application that helps users discover movies they might enjoy. The project follows a structured workflow starting with data cleaning, data exploration, followed by modeling, findings, conclusions, recommendations and future steps.

# Dataset Overview
The project utilizes the MovieLens dataset, which includes:

**User Ratings:** User-provided ratings for various movies.
**Movie Details:** Metadata about movies, including titles, genres, and identifiers.
The dataset was preprocessed to address missing values, duplicates, and inconsistencies. The genres column, which contains multiple genres per movie, was exploded to create individual rows for each genre for more detailed content-based filtering.

# Exploratory Data Analysis (EDA)
EDA was conducted to:

Understand the distribution of user ratings and movie popularity.
Analyze the genre-wise rating trends and correlations.
Explore user behaviors, such as rating patterns and preferences.
Visualizations were generated to uncover insights into the dataset and inform the recommendation strategies.

# Recommendation Strategies
**Collaborative Filtering (CF):**
A model-based approach leveraging Singular Value Decomposition (SVD) to predict user preferences.
It analyzes patterns in user-movie interactions, identifying movies a user is likely to enjoy based on ratings from similar users or movies.
The system dynamically retrains the model when new user ratings are added, ensuring updated recommendations.

**Content-Based Filtering (CBF):**
Utilizes metadata about movies, such as genres, to recommend movies similar to those a user has enjoyed.
The genres column is exploded, allowing the system to focus on individual genres for more precise similarity calculations.
Cosine similarity is computed between movies based on their genre encoding, enabling personalized suggestions rooted in content relevance.

**Hybrid Recommendations:**
Combines the strengths of Collaborative Filtering and Content-Based Filtering to enhance recommendation quality.
Hybrid scores are calculated using a weighted average of CF and CBF scores, with a tunable parameter alpha to control the weight (default: 0.7 for CF and 0.3 for CBF).
This approach balances personalization with content relevance, resulting in recommendations that align with both user preferences and movie characteristics.

**Key Features**
Personalized movie recommendations tailored to user ratings.
Option to filter recommendations by genre.
Dynamic model retraining for updated recommendations when new user data is added.
An intuitive hybrid scoring mechanism combining collaborative and content-based filtering.

# Installation and Usage
Clone the repository.
Install the required Python libraries using:
pip install -r requirements.txt or install the libraries listed below.
Run the Jupyter notebooks or Python scripts for:
Exploratory Data Analysis (EDA)
Model training and evaluation
Generating movie recommendations

# Future Enhancements
Incorporate temporal data, such as release dates, for time-aware recommendations.
Explore advanced models like neural collaborative filtering.
Extend the system to handle multi-user recommendations in real-time.

## TOOLS AND TECHNOLOGIES USED:
***

# Standard Data Science packages: 
#Importing the necessary libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise.model_selection import cross_validate,GridSearchCV
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline,SVD
from scipy.stats import f_oneway
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from surprise import SVD,Reader, Dataset
from surprise.prediction_algorithms.knns import KNNBasic
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import Functions
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


## DATASETS:
The datasets used in this project are in the file named Data. 

## PROJECT REPORT:
There is also a PDF file in this repository that is a report of the entire project including it's findings and recommendations.

## PRESENTATION SLIDES
There are presentation slides with a summary of the entire process


