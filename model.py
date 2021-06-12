#!/usr/bin/env python
# coding: utf-8

# In[3]:
my_api_key = 'b528863227d1c7bd8902923cce47e267'

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

# In[4]:


#FUNCTIONS TO GET MOVIE TITLE FROM THE INDEX AND VICE-VERSA

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    #try:
    return df[df.title.map(lambda x: x.lower()) == title.lower()]["index"].values[0]
    # except:
    #     return "No such movie found, check the keywords correctly" lskd


# In[5]:


#LOAD DATASET
df = pd.read_csv("./movie_dataset.csv")
parameters = ["keywords","cast","genres","director"]

#FILLING THE NA PLACES
for parameter in parameters:
    df[parameter] = df[parameter].fillna(" ")
    
#NEW COLUMN CONTAINING THE FINAL PARAMETERS
def final_parameter(row):
    return row["keywords"] + " " +row["cast"]+row["genres"]+" "+row["director"]
df["final_parameter"] = df.apply(final_parameter, axis = 1)
#print(df.iloc[0])
df["title"] = df["title"].fillna(" ")
def get_details_using_api(row):
    try:
        r = requests.get('https://api.themoviedb.org/3/search/movie?api_key='+my_api_key+'&query='+row["title"])
        s = json.loads(r.text)
        if len(s["results"]) >= 1:
            return s["results"][0]["poster_path"]
        else:
            return "No image for such movies"
    except:
        print(s)
        print(row["title"])
# print(df["title"].head())
df["poster_path"] = df.apply(get_details_using_api, axis = 1)
# print(df.loc[0,"poster_path"])
print(df.shape(0))
#print(df["title"])
#CONVERTING INTO VECTORS
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["final_parameter"])

#USING COSINE SIMILARITY ON THE VECTORS
similarity_matrix = cosine_similarity(count_matrix)


# In[6]:

def listofmovies(movie):
    #MOVIE TITLE ENTERED BY THE USER
    movie_entered_by_user = movie
    try:
        movie_index = get_index_from_title(movie_entered_by_user)
    except:
        return 'No such movie exist, increase your knowledge'
    # try:
    #     movie_index = get_index_from_title(movie_entered_by_user)
    # except:
    #     movie_index = "lksdjklfje"
    #print(movie_index)

    #PRINTING THE TOP 50 SIMILAR MOVIES USING SIMILARITY_MATRIX
    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1],reverse=True)
    # print(similar_movies)
    # print("===========================================================")
    # print(sorted_similar_movies)
    movieslist = []
    count = 0
    for movie in sorted_similar_movies:
            movieslist.append(get_title_from_index(movie[0]))
            count += 1
            if count >= 50:
                break
    return movieslist



# In[ ]:





# In[ ]:




