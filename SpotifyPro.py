# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:24:28 2020

@author: Talib
"""
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import seaborn as sns


#Getting Data from Spotify API

CLIENT_ID = 'ab18d1a54c0f4efa8d5532865c0be364'
CLIENT_SECRET = '98f3a7fcc17b4f22b6ef0dd7d8fa3a67'

PLAYLIST_ID = '37i9dQZF1DWYJ5kmTbkZiz'

client_credentials_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

tracks = sp.user_playlist_tracks(user = 'Spotify' , playlist_id = PLAYLIST_ID)
tracks_uri_list = [x['track']['uri'] for x in tracks['items']]

features = []
for i in tracks_uri_list:
    features = features + sp.audio_features(i)

features_df = pd.DataFrame(features)
cols_to_drop = ['id','analysis_url','key','time_signature','track_href','uri','mode','type','duration_ms']
features_df = features_df.drop(cols_to_drop,axis = 1)

#Using the Data Training the model

from sklearn.cluster import KMeans

#Scaling

for col in ['loudness','tempo']:
    features_df[col] = (features_df[col] - features_df[col].min())/(features_df[col].max()-features_df[col].min())
    
#Cluster size setection using Score Test of various size clusters
    
score_list = []
no_of_clusters = []

for i in range(2,10):
    no_of_clusters.append(i)
    kmeans_model = KMeans(n_clusters = i,random_state = 3).fit(features_df)
    preds = kmeans_model.predict(features_df)
    score_list.append(kmeans_model.inertia_)
    
df = pd.DataFrame(list(zip(no_of_clusters,score_list)),columns = ["Clusters","Score"])
    
sns.lineplot(x = "Clusters",y = "Score",data = df)

kmeans_model_final = KMeans(n_clusters = 5,random_state = 3).fit(features_df)
preds_final = kmeans_model_final.predict(features_df)


songs = []
artists = []

for i in tracks['items']:
    songs.append(i['track']['name'])
    artists.append(i['track']['artists'])
    
artists_list = []

for group in artists:
    artist_group = []
    for individual in group:
        artist_group.append(individual['name'])
    artists_list.append(' ,'.join(artist_group))

features_df['cluster'] = preds_final
   

clusters = features_df.groupby('cluster').agg('mean')

features_df['Songs'] = songs
features_df['Artists'] = artists_list

res = features_df.filter(['Songs','Artists','cluster'])

for i in range(0,5):
    print(res[res.cluster==i])
    
