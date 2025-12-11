import os
import json
import requests
import yt_dlp
import openl3
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY', '')
OUTPUT_DIR = 'song_database'
AUDIO_DIR = os.path.join(OUTPUT_DIR, 'audio')
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, 'embeddings.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


def get_top_songs(limit=100):
    print(f"Fetching top {limit} songs from Last.fm...")
    songs = []
    page = 1

    while len(songs) < limit:
        url = f'https://ws.audioscrobbler.com/2.0/?method=chart.gettoptracks&api_key={LASTFM_API_KEY}&format=json&page={page}&limit=50'
        response = requests.get(url)
        data = response.json()

        if 'tracks' not in data or 'track' not in data['tracks']:
            break

        for track in data['tracks']['track']:
            songs.append({
                'title': track['name'],
                'artist': track['artist']['name'],
                'playcount': int(track['playcount'])
            })

            if len(songs) >= limit:
                break

        page += 1
        time.sleep(0.2)

    return songs[:limit]


def download_song(song, output_path):
    query = f"{song['artist']} {song['title']} official audio"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.mp3', ''),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"ytsearch1:{query}"])
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False


def extract_openl3_embedding(audio_path):
    audio, sr = sf.read(audio_path)
    emb, ts = openl3.get_audio_embedding(
        audio,
        sr,
        content_type='music',
        embedding_size=512
    )
    avg_emb = np.mean(emb, axis=0)
    return avg_emb.tolist()


def extract_librosa_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30.0, sr=22050, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    tempo = float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    chroma_vals = np.sum(chroma, axis=1)
    chroma_vals = chroma_vals / np.sum(chroma_vals)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)

    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    max_corr = -1
    detected_key = 'C'
    detected_mode = 'Major'

    for i in range(12):
        major_rotated = np.roll(major_profile, i)
        minor_rotated = np.roll(minor_profile, i)

        major_corr = np.correlate(chroma_vals, major_rotated)[0]
        minor_corr = np.correlate(chroma_vals, minor_rotated)[0]

        if major_corr > max_corr:
            max_corr = major_corr
            detected_key = keys[i]
            detected_mode = 'Major'

        if minor_corr > max_corr:
            max_corr = minor_corr
            detected_key = keys[i]
            detected_mode = 'Minor'

    rms = librosa.feature.rms(y=y, hop_length=512)
    energy = float(np.mean(rms))

    return {
        'tempo': round(tempo, 1),
        'key': detected_key,
        'mode': detected_mode,
        'energy': round(energy, 4)
    }


def main():
    if not LASTFM_API_KEY:
        print("Error: LASTFM_API_KEY not set in .env file")
        return

    songs = get_top_songs(limit=100)
    print(f"Retrieved {len(songs)} songs\n")

    embeddings_db = {}

    for idx, song in enumerate(tqdm(songs, desc="Processing songs")):
        song_id = f"{idx:04d}"
        audio_path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")

        if not os.path.exists(audio_path):
            if not download_song(song, audio_path):
                continue

        if not os.path.exists(audio_path):
            continue

        try:
            embedding = extract_openl3_embedding(audio_path)
            features = extract_librosa_features(audio_path)

            embeddings_db[song_id] = {
                'title': song['title'],
                'artist': song['artist'],
                'playcount': song['playcount'],
                'rank': idx + 1,
                'embedding': embedding,
                'tempo': features['tempo'],
                'key': features['key'],
                'mode': features['mode'],
                'energy': features['energy']
            }
        except Exception as e:
            print(f"\nFailed to process {song['artist']} - {song['title']}: {e}")

    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(embeddings_db, f, indent=2)

    print(f"\n\nDatabase built successfully!")
    print(f"Processed {len(embeddings_db)} songs")
    print(f"Saved to {EMBEDDINGS_FILE}")


if __name__ == '__main__':
    main()
