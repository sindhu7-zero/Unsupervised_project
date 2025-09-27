import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# ====== 1️⃣ Load dataset ======
# 🔸 Update this path to your actual file
file_path = r'C:\Users\admin\Documents\Songs\Unsupervised_project\single_genre_artists.csv'

if not os.path.exists(file_path):
    st.error("❌ Dataset not found! Please check the file path.")
    st.stop()

df = pd.read_csv(file_path)

# Ensure required columns exist
required_cols = ['danceability','energy','loudness','speechiness',
                 'acousticness','instrumentalness','liveness','valence',
                 'tempo','duration_ms', 'name_song']
if not all(col in df.columns for col in required_cols):
    st.error("❌ Missing required columns in dataset.")
    st.stop()

# ====== 2️⃣ Prepare features ======
selected_features = ['danceability','energy','loudness','speechiness',
                     'acousticness','instrumentalness','liveness','valence',
                     'tempo','duration_ms']

df_features = df[selected_features + ['name_song']].copy().dropna()

# ====== 3️⃣ Standardize & sample ======
scaler = StandardScaler()
X = scaler.fit_transform(df_features[selected_features])
X = pd.DataFrame(X, columns=selected_features)

# Sample for speed (20k rows)
X_small = X.sample(min(20000, len(X)), random_state=42)
df_features_small = df_features.loc[X_small.index].reset_index(drop=True)
X_small = X_small.reset_index(drop=True)

# ====== 4️⃣ Compute similarity ======
cos_sim_small = cosine_similarity(X_small)

# ====== 5️⃣ Recommendation function ======
def recommendation(song_index, top_n=3):
    scores = list(enumerate(cos_sim_small[song_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != song_index]  # exclude self
    scores = scores[:top_n]
    return [df_features_small.iloc[i]['name_song'] for i, _ in scores]

# ====== 6️⃣ Streamlit UI ======
st.set_page_config(page_title="🎵 Smart Song Recommender", layout="centered")

# Custom CSS for better look
st.markdown("""
<style>
    .main-header { text-align: center; color: #1e88e5; }
    .recommend-card {
        background-color: #f9f9ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #4caf50;
    }
    .footer { text-align: center; font-size: 14px; color: #777; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>✨ Smart Song Recommender</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='text-align: center; color: #555;'>
Find songs that *feel* like your favorite — based on audio features like energy, mood, and rhythm!
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Settings")
    selected_song = st.selectbox(
        "Pick a song you love:",
        df_features_small['name_song'].tolist(),
        index=0
    )
    top_n = st.slider("How many recommendations?", 1, 10, 3)

# Main button
if st.button("🎧 Get Recommendations", use_container_width=True):
    try:
        song_index = df_features_small[df_features_small['name_song'] == selected_song].index[0]
        results = recommendation(song_index, top_n=top_n)

        st.success(f"✅ Based on **{selected_song}**, you might love these:")
        
        # Display results in styled cards
        for song in results:
            emoji = random.choice(['🎵', '🎧', '🔥', '💫', '🎹', '💿'])
            st.markdown(f"""
            <div class="recommend-card">
                <h4>{emoji} {song}</h4>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ Something went wrong. Please try another song.")

# Explain how it works
with st.expander("🔍 How does this work?"):
    st.write("""
    - This is a **content-based recommender system**.
    - It uses **audio features** (like energy, valence, tempo) from songs.
    - **Cosine similarity** measures how "aligned" two songs are in feature space — not physical distance.
    - Unlike Manhattan or Euclidean distance, cosine similarity focuses on *angle*, not magnitude — great for comparing song "vibes"!
    - Built with **scikit-learn** and **Streamlit** for a clean, interactive experience.
    """)

st.markdown('<div class="footer">Made with ❤️ | Unsupervised Learning Project</div>', unsafe_allow_html=True)