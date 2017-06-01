#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 07 12:49:55 2017

@author: hardeepsingh
"""

# In[1]:
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from time import time
import os
import urllib
import zipfile

conf = SparkConf().setAppName('appName').setMaster('local')
sc = SparkContext(conf=conf)

# Specify Dataset URL and Download Path
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
datasets_path = os.path.join('/Users/hardeepsingh/Documents/Classes/CS_599/FinalProject', 'datasets')
# In[2]:
    
# Download data using url's 
complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
small_f = urllib.urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urllib.urlretrieve (complete_dataset_url, complete_dataset_path) 

# Extract data into the same directory
with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)

with zipfile.ZipFile(complete_dataset_path, "r") as z:
    z.extractall(datasets_path)
    
# In[3]:
    
# Load the complete 'Rating' dataset file
complete_ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

# Parse
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line=complete_ratings_raw_data_header)
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
    
print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

# In[4]

# Load the complete 'Movies' dataset file
complete_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)    
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))
    
print "There are %s movies in the complete dataset" % (complete_movies_titles.count())

# In[5]:

# Get Counts and Averages on Rating Predictions
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0])) 

# In[6]:
    
# Adding new user ratings
new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
     (0,260,9), # Star Wars (1977)
     (0,1,8), # Toy Story (1995)
     (0,16,7), # Casino (1995)
     (0,25,8), # Leaving Las Vegas (1995)
     (0,32,9), # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
     (0,335,4), # Flintstones, The (1994)
     (0,379,3), # Timecop (1994)
     (0,296,7), # Pulp Fiction (1994)
     (0,858,10) , # Godfather, The (1972)
     (0,50,8) # Usual Suspects, The (1995)
    ]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print 'New user ratings: %s' % new_user_ratings_RDD.take(10)

# Append to Complete Rating Dataset
complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
# In[7]:

# Train ALS model
seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]

t0 = time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, ranks[2], seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
tt = time() - t0

print "New model trained in %s seconds" % round(tt,3)
# In[8]:
    
# Getting top recommendations  
new_user_ratings_ids = map(lambda x: x[1], new_user_ratings) 

# Keep movies that are not part of user list
new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0])))

# Using train model weights make a prediction call
new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
# In[9]:

# Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD =     new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
new_user_recommendations_rating_title_and_count_RDD.take(10)
# In[10]:
    
# Convert format to Title, Rating, Rating Count
new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[0], r[1][0][1], r[1][0][0], r[1][1]))
new_user_recommendations_rating_title_and_count_RDD.take(10)
# In[11]:
    
# Finally, filter the rating greater than or equal to 7
top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=7).takeOrdered(10, key=lambda x: -x[3])
print ('TOP recommended movies (with more than 10 reviews):\n%s' % '\n'.join(map(str, top_movies)))

# In[12]:
    
# Get movie Posters
import imdb
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

access = imdb.IMDb()

# Get List of Recommended Movie ID's
recommended_movie_ids = map(lambda x: x[0], top_movies)

# Read Links.csv file to get all IMDB ID's using current movie ID's
df = pd.read_csv(os.path.join(datasets_path, 'ml-latest', 'links.csv'))
df = df[df['movieId'].isin(recommended_movie_ids)]
recommended_imdb_ids = df['imdbId'].tolist()

# Show movie poster with title
for movie in recommended_imdb_ids:
    movie = access.get_movie(movie)
    print "Title: %s (%s) \nRating on IMDB : %s" % (movie['title'], movie['year'], movie['rating'])
    f = urllib.urlopen(movie['cover url'])
    im = Image.open(f)
    plt.imshow(im)
    plt.show()
