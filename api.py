#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json


# In[2]:


df = pd.read_csv("./movie_dataset_processed.csv")


# In[3]:


df.columns


# In[4]:


def get_data_from_index(index):
    return {"title":df[df.index == index]["title"].values[0],"poster_path":df[df.index == index]["poster_path"].values[0],"director":df[df.index == index]["director"].values[0],"cast":df[df.index == index]["cast"].values[0],"overview":df[df.index == index]["overview"].values[0],"homepage":df[df.index == index]["homepage"].values[0],"production_companies":df[df.index == index]["production_companies"].values[0]}
def get_index_from_title(title):
    #try:
    return df[df.title.map(lambda x: x.lower()) == title.lower()]["index"].values[0]


# In[5]:


#CONVERTING INTO VECTORS
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["final_parameter"])

#USING COSINE SIMILARITY ON THE VECTORS
similarity_matrix = cosine_similarity(count_matrix)


# In[6]:


parameters = ["title","cast","director","overview","homepage","production_companies"]
for parameter in parameters:
    df[parameter] = df[parameter].fillna(" ")
df["poster_path"] = df["poster_path"].fillna("/svYyAWAH3RThMmHcCaJZ97jnTtT.jpg")


# In[7]:


#MOVIE TITLE ENTERED BY THE USER
def listofmovies(movie):
    data = [{'title': 'Inception', 'poster_path': '/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg', 'director': 'Christopher Nolan', 'cast': 'Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page Tom Hardy Ken Watanabe', 'overview': 'Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets is offered a chance to regain his old life as payment for a task considered to be impossible: "inception", the implantation of another person\'s idea into a target\'s subconscious.', 'homepage': 'http://inceptionmovie.warnerbros.com/', 'production_companies': '[{"name": "Legendary Pictures", "id": 923}, {"name": "Warner Bros.", "id": 6194}, {"name": "Syncopy", "id": 9996}]'},{'title': 'Harry Potter and the Half-Blood Prince', 'poster_path': '/7v6mwWY8fIlEuib37QMmZGRYCL8.jpg', 'director': 'David Yates', 'cast': 'Daniel Radcliffe Rupert Grint Emma Watson Tom Felton Michael Gambon', 'overview': "As Harry begins his sixth year at Hogwarts, he discovers an old book marked as 'Property of the Half-Blood Prince', and begins to learn more about Lord Voldemort's dark past.", 'homepage': 'http://harrypotter.warnerbros.com/harrypotterandthehalf-bloodprince/dvd/index.html', 'production_companies': '[{"name": "Warner Bros.", "id": 6194}, {"name": "Heyday Films", "id": 7364}]'},{'title': 'Men in Black 3', 'poster_path': '/90DdoEStzeObs96fsYf4GG544iN.jpg', 'director': 'Barry Sonnenfeld', 'cast': 'Will Smith Tommy Lee Jones Josh Brolin Michael Stuhlbarg Emma Thompson', 'overview': "Agents J (Will Smith) and K (Tommy Lee Jones) are back...in time. J has seen some inexplicable things in his 15 years with the Men in Black, but nothing, not even aliens, perplexes him as much as his wry, reticent partner. But when K's life and the fate of the planet are put at stake, Agent J will have to travel back in time to put things right. J discovers that there are secrets to the universe that K never told him - secrets that will reveal themselves as he teams up with the young Agent K (Josh Brolin) to save his partner, the agency, and the future of humankind.", 'homepage': 'http://www.sonypictures.com/movies/meninblack3/', 'production_companies': '[{"name": "Amblin Entertainment", "id": 56}, {"name": "Media Magik Entertainment", "id": 5627}, {"name": "Imagenation Abu Dhabi FZ", "id": 6736}, {"name": "Hemisphere Media Capital", "id": 9169}, {"name": "Parkes/MacDonald Productions", "id": 11084}]'},{'title': 'Jurassic World', 'poster_path': '/rhr4y79GpxQF9IsfJItRXVaoGs4.jpg', 'director': 'Colin Trevorrow', 'cast': "Chris Pratt Bryce Dallas Howard Irrfan Khan Vincent D'Onofrio Nick Robinson", 'overview': 'Twenty-two years after the events of Jurassic Park, Isla Nublar now features a fully functioning dinosaur theme park, Jurassic World, as originally envisioned by John Hammond.', 'homepage': 'http://www.jurassicworld.com/', 'production_companies': '[{"name": "Universal Studios", "id": 13}, {"name": "Amblin Entertainment", "id": 56}, {"name": "Legendary Pictures", "id": 923}, {"name": "Fuji Television Network", "id": 3341}, {"name": "Dentsu", "id": 6452}]'},{'title': 'The Jungle Book', 'poster_path': '/xzpNa3RjKHfIijeqVuL72IgBvwy.jpg', 'director': 'Jon Favreau', 'cast': 'Neel Sethi Bill Murray Ben Kingsley Idris Elba Scarlett Johansson', 'overview': 'After a threat from the tiger Shere Khan forces him to flee the jungle, a man-cub named Mowgli embarks on a journey of self discovery with the help of panther, Bagheera, and free spirited bear, Baloo.', 'homepage': 'http://movies.disney.com/the-jungle-book-2016', 'production_companies': '[{"name": "Walt Disney Pictures", "id": 2}, {"name": "Walt Disney Studios Motion Pictures", "id": 3036}, {"name": "Fairview Entertainment", "id": 7297}, {"name": "Moving Picture Company (MPC)", "id": 20478}]'},{'title': 'The Amazing Spider-Man', 'poster_path': '/dQ8TOCYgP9pzQvSb1cmaalYqdb5.jpg', 'director': 'Marc Webb', 'cast': 'Andrew Garfield Emma Stone Rhys Ifans Denis Leary Campbell Scott', 'overview': "Peter Parker is an outcast high schooler abandoned by his parents as a boy, leaving him to be raised by his Uncle Ben and Aunt May. Like most teenagers, Peter is trying to figure out who he is and how he got to be the person he is today. As Peter discovers a mysterious briefcase that belonged to his father, he begins a quest to understand his parents' disappearance – leading him directly to Oscorp and the lab of Dr. Curt Connors, his father's former partner. As Spider-Man is set on a collision course with Connors' alter ego, The Lizard, Peter will make life-altering choices to use his powers and shape his destiny to become a hero.", 'homepage': 'http://www.theamazingspiderman.com', 'production_companies': '[{"name": "Columbia Pictures", "id": 5}, {"name": "Laura Ziskin Productions", "id": 326}, {"name": "Marvel Entertainment", "id": 7505}]'},{'title': 'The Lord of the Rings: The Return of the King', 'poster_path': '/rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg', 'director': 'Peter Jackson', 'cast': 'Elijah Wood Ian McKellen Viggo Mortensen Liv Tyler Orlando Bloom', 'overview': "Aragorn is revealed as the heir to the ancient kings as he, Gandalf and the other members of the broken fellowship struggle to save Gondor from Sauron's forces. Meanwhile, Frodo and Sam bring the ring closer to the heart of Mordor, the dark lord's realm.", 'homepage': 'http://www.lordoftherings.net', 'production_companies': '[{"name": "WingNut Films", "id": 11}, {"name": "New Line Cinema", "id": 12}]'},{'title': 'The Godfather', 'poster_path': '/3bhkrj58Vtu7enYsRolD1fZdja1.jpg', 'director': 'Francis Ford Coppola', 'cast': 'Marlon Brando Al Pacino James Caan Richard S. Castellano Robert Duvall', 'overview': 'Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family. When organized crime family patriarch, Vito Corleone barely survives an attempt on his life, his youngest son, Michael steps in to take care of the would-be killers, launching a campaign of bloody revenge.', 'homepage': 'http://www.thegodfather.com/', 'production_companies': '[{"name": "Paramount Pictures", "id": 4}, {"name": "Alfran Productions", "id": 10211}]'},{'title': 'The Shawshank Redemption', 'poster_path': '/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg', 'director': 'Frank Darabont', 'cast': 'Tim Robbins Morgan Freeman Bob Gunton Clancy Brown Mark Rolston', 'overview': 'Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope.', 'homepage': ' ', 'production_companies': '[{"name": "Castle Rock Entertainment", "id": 97}]'},{'title': 'Avatar', 'poster_path': '/6EiRUJpuoeQPghrs3YNktfnqOVh.jpg', 'director': 'James Cameron', 'cast': 'Sam Worthington Zoe Saldana Sigourney Weaver Stephen Lang Michelle Rodriguez', 'overview': 'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.', 'homepage': 'http://www.avatarmovie.com/', 'production_companies': '[{"name": "Ingenious Film Partners", "id": 289}, {"name": "Twentieth Century Fox Film Corporation", "id": 306}, {"name": "Dune Entertainment", "id": 444}, {"name": "Lightstorm Entertainment", "id": 574}]'},{'title': 'The Notebook', 'poster_path': '/rNzQyW4f8B8cQeg7Dgj3n6eT5k9.jpg', 'director': 'Nick Cassavetes', 'cast': 'Rachel McAdams Ryan Gosling Gena Rowlands James Garner Sam Shepard', 'overview': "An epic love story centered around an older man who reads aloud to a woman with Alzheimer's. From a faded notebook, the old man's words bring to life the story about a couple who is separated by World War II, and is then passionately reunited, seven years later, after they have taken different paths.", 'homepage': 'http://www.newline.com/properties/notebookthe.html', 'production_companies': '[{"name": "New Line Cinema", "id": 12}]'},{'title': 'The Perks of Being a Wallflower', 'poster_path': '/aKCvdFFF5n80P2VdS7d8YBwbCjh.jpg', 'director': 'Stephen Chbosky', 'cast': 'Logan Lerman Emma Watson Ezra Miller Paul Rudd Nina Dobrev', 'overview': 'A coming-of-age story based on the best-selling novel by Stephen Chbosky, which follows 15-year-old freshman Charlie, an endearing and naive outsider who is taken under the wings of two seniors. A moving tale of love, loss, fear and hope - and the unforgettable friends that help us through life.', 'homepage': 'http://perks-of-being-a-wallflower.com/', 'production_companies': '[{"name": "Summit Entertainment", "id": 491}, {"name": "Mr. Mudd Production", "id": 2130}]'},{'title': 'The Avengers', 'poster_path': '/RYMX2wcKCBAr24UyPD7xwmjaTn.jpg', 'director': 'Joss Whedon', 'cast': 'Robert Downey Jr. Chris Evans Mark Ruffalo Chris Hemsworth Scarlett Johansson', 'overview': 'When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency known as S.H.I.E.L.D., finds himself in need of a team to pull the world back from the brink of disaster. Spanning the globe, a daring recruitment effort begins!', 'homepage': 'http://marvel.com/avengers_movie/', 'production_companies': '[{"name": "Paramount Pictures", "id": 4}, {"name": "Marvel Studios", "id": 420}]'},{'title': 'The Dark Knight', 'poster_path': '/qJ2tW6WMUDux911r6m7haRef0WH.jpg', 'director': 'Christopher Nolan', 'cast': 'Christian Bale Heath Ledger Aaron Eckhart Michael Caine Maggie Gyllenhaal', 'overview': 'Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker.', 'homepage': 'http://thedarkknight.warnerbros.com/dvdsite/', 'production_companies': '[{"name": "DC Comics", "id": 429}, {"name": "Legendary Pictures", "id": 923}, {"name": "Warner Bros.", "id": 6194}, {"name": "DC Entertainment", "id": 9993}, {"name": "Syncopy", "id": 9996}]'},{'title': '12 Angry Men', 'poster_path': '/ppd84D2i9W8jXmsyInGyihiSyqz.jpg', 'director': 'Sidney Lumet', 'cast': 'Henry Fonda Martin Balsam John Fiedler Lee J. Cobb E.G. Marshall', 'overview': "The defense and the prosecution have rested and the jury is filing into the jury room to decide if a young Spanish-American is guilty or innocent of murdering his father. What begins as an open and shut case soon becomes a mini-drama of each of the jurors' prejudices and preconceptions about the trial, the accused, and each other.", 'homepage': ' ', 'production_companies': '[{"name": "United Artists", "id": 60}, {"name": "Orion-Nova Productions", "id": 10212}]'}]
    if movie == "homepage":
        return data
    try:
        movie_index = get_index_from_title(movie)
    except:
        return [{"title":"Can't find the movie :( , try with correct keywords or some different movie."}]

    #PRINTING THE TOP 50 SIMILAR MOVIES USING SIMILARITY_MATRIX
    
    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1],reverse=True)
    # print(similar_movies)
    # print("===========================================================")
    # print(sorted_similar_movies)
    
    
    movieslist = []
    count = 0
    for movie in sorted_similar_movies:
            movieslist.append(get_data_from_index(movie[0]))
            count += 1
            if count >= 50:
                break
    return movieslist


# In[ ]:





# In[36]:


#df.to_csv("./movie_dataset_processed.csv")


# In[ ]:




