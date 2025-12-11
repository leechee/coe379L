import requests
import sys

def upload_audio(file_path):
    url = 'http://localhost:5000/analyze'

    with open(file_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        print("=== Analysis Results ===\n")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"\n=== Audio Features ===")
        print(f"Tempo: {result['features']['tempo']} BPM")
        print(f"Key: {result['features']['key']} {result['features']['mode']}")
        print(f"Energy: {result['features']['energy']}")
        print(f"Brightness: {result['features']['brightness']}")

        print(f"\n=== Top 10 Similar Songs ===")
        for i, song in enumerate(result['recommendations'], 1):
            print(f"{i}. {song['artist']} - {song['title']}")
            print(f"   Similarity: {song['similarity_score']:.3f}")
            if song.get('tempo'):
                print(f"   Tempo: {song['tempo']} BPM, Key: {song['key']} {song['mode']}")
            print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_upload.py <audio_file.mp3>")
        sys.exit(1)

    upload_audio(sys.argv[1])
