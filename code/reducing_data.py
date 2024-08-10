import pandas as pd
from collections import Counter

ratings_df = pd.read_csv('ml-20m/ratings.csv')

# Decreasing index by 1 as we will need it later for MF
ratings_df['userId'] = ratings_df['userId'] - 1

# Mapping movies to have continous id´s
movie2idx = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}
ratings_df['movie_idx'] = ratings_df['movieId'].map(movie2idx)

# Eliminating timestamp column as it is not important
ratings_df = ratings_df.drop(columns=['timestamp'])

ratings_df.to_csv('edited_rating.csv', index=False)

df = pd.read_csv('edited_rating.csv')

# Limiting the size of the dataframe to have only the most frequent movies and users
N = df.userId.max() + 1
M = df.movie_idx.max() + 1

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

n = 15000  # Top n users
m = 3000   # Top m movies

user_ids = [n for n, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]

df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

# Indexing movies and user ids to the new reduced subset
def restart_index(df, column):
    unique_values = df[column].unique()
    id_mapping = {value: idx for idx, value in enumerate(unique_values)}
    df[column] = df[column].map(id_mapping)
    return df

df_small = restart_index(df_small, 'userId')
df_small = restart_index(df_small, 'movie_idx')

#Saving the dataframe
df_small.to_csv('reduced_rating.csv', index=False)

print(df_small.head(10))