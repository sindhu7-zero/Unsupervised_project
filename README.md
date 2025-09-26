Amazon Music Clustering –Project Report
1. Introduction
The objective of this project was to cluster songs from the Amazon Music dataset based on their audio features, using unsupervised machine learning techniques. With millions of songs available, it is impractical to manually categorize them into genres or moods. By analyzing features such as dance ability, energy, loudness, and tempo, this project aimed to automatically group songs into meaningful clusters and further build a simple recommendation system.
2. Dataset Overview
•	Dataset used: single_genre_artists.csv
•	Shape: ~95,000 rows × 18 columns
Initial Exploration
•	Checked dataset structure using .info()
•	Verified datatypes of all features
•	Counted null values → no significant missing data
•	Checked for duplicates → some duplicate rows were found and dropped
Dropped Columns
•	Non-numeric and non-relevant fields such as id_songs, id_artists, artist_name, release_date were excluded from clustering.
•	Kept only audio-related features.
Feature Selection & Scaling
Selected features that describe the sound characteristics of songs:
•	danceability
•	energy
•	loudness
•	speechiness
•	acousticness
•	instrumentalness
•	liveness
•	valence
•	tempo
•	duration_ms
Since clustering is distance-based, features were scaled using StandardScaler to ensure equal weight for all attributes.
4. Clustering Approach
4.1 KMeans Clustering
•	Applied KMeans algorithm on the scaled dataset.
•	Tested number of clusters k in the range 1–25.
4.2 Elbow Method
•	Plotted WCSS (Within-Cluster Sum of Squares) vs. number of clusters.
•	Observed the "elbow point" around k = 5.
4.3 Silhouette Score
•	Calculated silhouette scores for different values of k.
•	Best balance of compactness and separation was also around k = 5.
4.4 Final Model
•	Chose KMeans with 5 clusters.
•	Added new column Target to store cluster assignments for each song.
5. Evaluation
•	Inertia (WCSS): indicated good cluster compactness.
•	Silhouette Score: positive (>0.3), showing clusters are reasonably well separated.
•	Cluster Sizes: not perfectly balanced, but interpretable.
Cluster Profiling
By comparing average feature values in each cluster:
•	Cluster 0: high acousticness, low energy → acoustic/relaxing tracks
•	Cluster 1: high danceability, moderate energy → pop/party songs
•	Cluster 2: high energy, loudness → rock/electronic type tracks
•	Cluster 3: speech-heavy songs → rap/spoken word
•	Cluster 4: mixed attributes, experimental tracks
6. Recommendation System 
After clustering, I extended the project to build a song recommendation system.
Approach
•	Used cosine similarity between feature vectors.
•	Due to memory limitations, instead of computing similarity for all ~95k songs, I sampled 20,000 songs.
•	Built a function to recommend top-N similar songs for a given track.
Streamlit App
•	Developed a simple Streamlit interface where a user can:
o	Select a song from a dropdown.
o	Choose number of recommendations.
o	Get the most similar songs with similarity scores.
This made the system more interactive and closer to a real-world application.
7. Business Use Cases
1.	Personalized Playlist Curation: Grouping songs automatically into mood-based playlists.
2.	Improved Song Discovery: Suggesting new tracks to users similar to their favorites.
3.	Artist Analysis: Helping artists understand competitive positioning of their tracks.
4.	Market Segmentation: Helping streaming platforms analyze listening patterns.
8. Results & Conclusion
•	Successfully clustered ~95k songs into 5 meaningful groups using KMeans.
•	Achieved reasonable silhouette scores and interpretable clusters.
•	Implemented an additional recommendation system using cosine similarity.
•	Built a Streamlit app to showcase song similarity search.
9. Skills Gained
•	Data exploration & cleaning
•	Feature engineering & scaling
•	KMeans clustering
•	Elbow method & silhouette score evaluation
•	Recommendation system design
•	Streamlit app development
•	Data storytelling & cluster interpretation
