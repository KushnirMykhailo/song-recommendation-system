from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from lyricsgenius import Genius
import re
import base64
import requests

SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""
genius = Genius("", skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"])

app = Flask(__name__)

# Завантаження моделі і даних
df = pd.read_csv(r"C:\d\python\ling\datasets\labeled_lyrics_dataset.csv")
df2 = pd.read_csv(r"C:\d\python\ling\datasets\recommendation_system_v1.csv")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['predicted_category'])

model_path = r"C:\d\python\ling\models\lyrics-bert-classifier_v2"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model.eval()

def get_spotify_token():
    auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    return response.json()["access_token"]


def clean_lyrics_text(text):
    # Якщо є "Read More", беремо все після неї
    if "Read More" in text:
        text = text.split("Read More", 1)[1]

    # Видаляємо секції типу [Intro], [Chorus: Name], [Verse 2], [Bridge], тощо
    text = re.sub(r"\[.*?\]", "", text)

    # Видаляємо зайві порожні рядки
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    return dict(zip(label_encoder.classes_, probs))

def recommend_similar_songs(input_lyrics, lyrics_df_with_probs, top_k=20):
    input_vector_dict = predict_proba(input_lyrics)
    input_vector = np.array([input_vector_dict[label] for label in label_encoder.classes_]).reshape(1, -1)
    prob_matrix = lyrics_df_with_probs[label_encoder.classes_].values
    similarities = cosine_similarity(input_vector, prob_matrix).flatten()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return input_vector_dict, lyrics_df_with_probs.iloc[top_k_indices].copy().assign(similarity=similarities[top_k_indices])

def recommend_by_single_category(input_lyrics, lyrics_df_with_probs, category, top_k=20):
    input_vector_dict = predict_proba(input_lyrics)
    if category not in input_vector_dict:
        return pd.DataFrame()
    input_val = input_vector_dict[category]
    song_vals = lyrics_df_with_probs[category].values
    similarities = -np.abs(song_vals - input_val)  # Closer = better
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    return lyrics_df_with_probs.iloc[top_k_indices].copy().assign(similarity=-np.abs(song_vals[top_k_indices] - input_val))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    input_lyrics = data["lyrics"]
    input_probs, recommendations = recommend_similar_songs(input_lyrics, df2)
    return jsonify({
        "input_probs": input_probs,
        "recommendations": recommendations.to_dict(orient="records")
    })

@app.route("/recommend-category", methods=["POST"])
def recommend_category():
    data = request.get_json()
    input_lyrics = data["lyrics"]
    category = data["category"]
    recommendations = recommend_by_single_category(input_lyrics, df2, category)
    return jsonify({
        "recommendations": recommendations.to_dict(orient="records")
    })

@app.route("/fetch-lyrics", methods=["POST"])
def fetch_lyrics():
    data = request.get_json()
    title = data.get("title", "")
    artist = data.get("artist", "")
    if not title or not artist:
        return jsonify({"error": "Missing title or artist"}), 400

    try:
        song = genius.search_song(title=title, artist=artist)
        lyrics = song.lyrics if song else None
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"lyrics": clean_lyrics_text(lyrics)})


@app.route("/fetch-lyrics-from-spotify", methods=["POST"])
def fetch_lyrics_from_spotify():
    data = request.get_json()
    url = data.get("spotify_url", "")

    match = re.search(r"track/([a-zA-Z0-9]+)", url)
    if not match:
        return jsonify({"error": "Invalid Spotify URL"}), 400

    track_id = match.group(1)
    try:
        token = get_spotify_token()
        headers = {"Authorization": f"Bearer {token}"}
        track_res = requests.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers)
        track_res.raise_for_status()
        track_info = track_res.json()

        title = track_info["name"]
        artist = track_info["artists"][0]["name"]

        song = genius.search_song(title=title, artist=artist)
        lyrics = clean_lyrics_text(song.lyrics) if song else None

        return jsonify({"lyrics": lyrics, "title": title, "artist": artist})

    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- Додай це
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
