import requests
import sys

def upload_audio(file_path):
    url = 'http://localhost:5000/analyze'

    print(f"Uploading {file_path}...")
    print("Processing (this may take 2-5 minutes on first run while downloading OpenL3 models)...")

    with open(file_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(url, files=files, timeout=600)

    if response.status_code == 200:
        result = response.json()

        # Check if the response indicates success
        if not result.get('success', False):
            print(f"Error from server: {result.get('error', 'Unknown error')}")
            return

        features = result.get('features', {})
        duration = result.get('duration', 0)

        print("\n=== Analysis Results ===\n")
        print(f"Duration: {duration:.2f} seconds")

        print(f"\n=== Audio Features ===")
        print(f"Tempo: {features.get('tempo', 'N/A')} BPM")
        print(f"Key: {features.get('key', 'N/A')} {features.get('mode', 'N/A')}")
        print(f"Energy: {features.get('energy', 'N/A')}")
        print(f"Brightness: {features.get('brightness', 'N/A')}")

        similar_songs = result.get('recommendations', [])

        if not similar_songs:
            print(f"\n=== No Similar Songs Found ===")
            print("This might mean:")
            print("- The song database is empty")
            print("- The database songs don't have the required features")
            print("- There was an error in similarity calculation")
        else:
            print(f"\n=== Top 10 Similar Songs ===")
            for i, song in enumerate(similar_songs[:10], 1):
                print(f"{i}. {song['artist']} - {song['title']}")
                print(f"   Similarity: {song['similarity_score']:.3f}")
                print(f"   Tempo: {song.get('tempo', 'N/A')} BPM, Key: {song.get('key', 'N/A')} {song.get('mode', 'N/A')}")
                print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <audio_file.mp3>")
        sys.exit(1)

    upload_audio(sys.argv[1])
