import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
import pickle

df_small = pd.read_csv('reduced_rating.csv')

# Creting user2movie, movie2user y usermovie2rating dictionaries for MF
user2movie = {}
movie2user = {}
usermovie2rating = {}

def update_user2movie_and_movie2user(row):
    i = int(row.userId)
    j = int(row.movie_idx)
    
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)

    usermovie2rating[(i, j)] = row.rating

df_small.apply(update_user2movie_and_movie2user, axis=1)

# Converting user2movie y movie2user to include ratinfs
user2movierating = {i: (movies, np.array([usermovie2rating[(i, j)] for j in movies])) for i, movies in user2movie.items()}
movie2userrating = {j: (users, np.array([usermovie2rating[(i, j)] for i in users])) for j, users in movie2user.items()}

# Divide our dataset in test and training
df_small = shuffle(df_small)
cutoff = int(0.8 * len(df_small))
df_train = df_small.iloc[:cutoff]
df_test = df_small.iloc[cutoff:]

# We created usermovie2rating_test to test :v
usermovie2rating_test = {}
df_test.apply(lambda row: usermovie2rating_test.update({(int(row.userId), int(row.movie_idx)): row.rating}), axis=1)

# Creating movie2userrating_test to test as well
movie2userrating_test = {}
for (i, j), r in usermovie2rating_test.items():
    if j not in movie2userrating_test:
        movie2userrating_test[j] = [[i], [r]]
    else:
        movie2userrating_test[j][0].append(i)
        movie2userrating_test[j][1].append(r)

# Converting list of ratings in numpy arrays
for j, (users, r) in movie2userrating_test.items():
    movie2userrating_test[j][1] = np.array(r)

# Saving the dictionaries to be used later
with open('user2movierating.pkl', 'wb') as f:
    pickle.dump(user2movierating, f)

with open('movie2userrating.pkl', 'wb') as f:
    pickle.dump(movie2userrating, f)

with open('usermovie2rating.pkl', 'wb') as f:
    pickle.dump(usermovie2rating, f)

with open('movie2userrating_test.pkl', 'wb') as f:
    pickle.dump(movie2userrating_test, f)
