import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# 1️⃣ Load dataset
df = pd.read_csv(r'C:\Users\admin\Documents\Songs\Unsupervised_project\single_genre_artists.csv')

# 2️⃣ Select features
selected_features = ['danceability','energy','loudness','speechiness',
                     'acousticness','instrumentalness','liveness','valence',
                     'tempo','duration_ms']

df_features = df[selected_features + ['name_song']].copy()

# 3️⃣ Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df_features[selected_features])
X = pd.DataFrame(X, columns=selected_features)

# 4️⃣ Sample for faster computation
X_small = X.sample(20000, random_state=42)
df_features_small = df_features.loc[X_small.index].reset_index(drop=True)
X_small = X_small.reset_index(drop=True)

# 5️⃣ Compute cosine similarity
cos_sim_small = cosine_similarity(X_small)

# 6️⃣ Recommendation function
def recommendation(song_index, top_n=3):
    scores = list(enumerate(cos_sim_small[song_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != song_index]
    scores = scores[:top_n]

    recommendations = [(df_features_small.iloc[i]['name_song'], round(score, 2)) for i, score in scores]
    return recommendations

# 7️⃣ Streamlit UI
st.title("Song Recommendation System 🎵")

# Select song
song_list = df_features_small['name_song'].tolist()
selected_song = st.selectbox("Choose a song:", song_list)

# Number of recommendations
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=3)

# Show recommendations
if st.button("Get Recommendations"):
    song_index = df_features_small[df_features_small['name_song'] == selected_song].index[0]
    results = recommendation(song_index, top_n=top_n)
    st.write("Top Recommendations:")
    for song, score in results:
        st.write(f"{song} (Similarity: {score})")
