import random
import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)

###############################################
# Data Preprocessing
###############################################

# Method for Matrix Transformation

def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] < 1000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

    return user_movie_df


movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_pickle("movie_lents_rating.pkl")
df = movie.merge(rating,how="left")
df.shape
df.head()

# Matrix in desired format
user_movie_df = create_user_movie_df(df)

###############################################
# User Based Recommendation
###############################################

###############################################
# Getting movies watched by the user to be recommended
###############################################
# Random user
from random import randint,seed
seed(5)
user = randint(1,len(user_movie_df))
# int(pd.Series(user_movie_df.index).sample(random_state=45).values)

user_df = user_movie_df[user_movie_df.index == user]
movies_watched = user_df.columns[user_df.notnull().any()].tolist() # İzledigi filmlerin listesi
len(movies_watched) # 19 film izlemis


###############################################
# Obtaning same users that rated same movies
###############################################
movies_watched_df = user_movie_df[movies_watched] # Userin izledigi filmerin olusturdugu df
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = pd.DataFrame(user_movie_count)
user_movie_count.rename( columns= { 0 :"movie_count"},inplace=True)
user_movie_count[user_movie_count["movie_count"] > 11].sort_values("movie_count", ascending=False)
# userin izledigi filmlerin %60'nı(en az 12 film) ni izleyenlerin idsini cekme
user_same_movies_id = user_movie_count[user_movie_count["movie_count"]>11]
len(user_same_movies_id) # 910 kisi

###############################################
# Getting users who are most similar to the user
###############################################

final_df = movies_watched_df[movies_watched_df.index.isin(user_same_movies_id.index)]
final_df.shape
final_df.T.corr()

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
#corr_df = corr_df.abs() negatif degerlere bakmak gerekir mi ?
top_users = corr_df[(corr_df["user_id_1"] == user) & (corr_df["corr"] >= 0.50)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
###############################################
# Calculation of Weighted Average Recommendation Score
###############################################
top_users_ratings["weighted_rating"] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings[top_users_ratings["userId"]==66964.0]
recommendation_df= top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.rename( columns= { "weighted_rating" :"Averag Weighted Rec Score"},inplace=True)

# User_based film recommendation
top_movies_to_be_recommend = recommendation_df.sort_values("Averag Weighted Rec Score",ascending=False).head(5)

top_movies_to_be_recommend = top_movies_to_be_recommend.merge(movie[["movieId", "title"]]) # Top 5 Films with title
top_movies_names_to_be_rec = top_movies_to_be_recommend[["title"]]

###############################################
# Item Based Recommendation
###############################################


# user = 66964

movie_id = rating[(rating["userId"] == user) & (rating["rating"]== 5.0)]. \
    sort_values("timestamp",ascending=False)["movieId"].values[0]

movie_name = movie.loc[movie["movieId"]== movie_id,"title"].values[0]
movie_name_df = user_movie_df[movie_name]
item_based_movie_reccommended = user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:6].index.tolist()
item_based_movie_reccommended_df = pd.DataFrame(item_based_movie_reccommended)

# Merging two recommendation system
final_recs = pd.concat([top_movies_names_to_be_rec,item_based_movie_reccommended_df],axis=1)
final_recs.columns = [["User_based_movies","Item_based_movies"]]
final_recs