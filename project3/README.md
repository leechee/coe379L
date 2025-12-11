# Music Recommendation System Using Audio Embeddings

A machine learning-based music recommendation system that analyzes audio files using OpenL3 embeddings and Librosa feature extraction to find similar songs.

## Overview

This project implements a content-based music recommendation system that uses deep learning and signal processing techniques to analyze audio similarity. The system extracts acoustic features from uploaded songs and compares them against a database of pre-processed tracks to find the most similar matches.

## Pipeline Architecture

The recommendation pipeline consists of two main components:

### 1. OpenL3 Audio Embeddings
- Pre-trained convolutional neural network trained on Google's AudioSet
- Converts audio waveforms into 512-dimensional embedding vectors
- Captures semantic audio features (timbre, rhythm, harmonic content)
- Similarity measured using cosine similarity between embedding vectors

### 2. Librosa Feature Extraction
- Tempo detection using beat tracking algorithms
- Key and mode detection using chroma features and Krumhansl-Schmuckler profiles
- Energy analysis using RMS (Root Mean Square)
- Spectral brightness using spectral centroid

### Recommendation Process
1. User uploads an audio file
2. System extracts OpenL3 embedding (512D vector) and Librosa features
3. Cosine similarity computed between upload and all database songs
4. Top 10 most similar songs returned with similarity scores

## Installation

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or pip

### Setup

1. Clone or download this project

2. Create a conda environment:
```bash
conda create -n music-rec python=3.10 -y
conda activate music-rec
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Last.fm API key:
   - Get a free API key at https://www.last.fm/api/account/create
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`

## Building the Song Database

Before running the system, you need to build a database of songs:

```bash
python build_database.py
```

This script will:
- Fetch the top 100 songs from Last.fm
- Download audio from YouTube
- Extract OpenL3 embeddings and Librosa features
- Save everything to `song_database/embeddings.json`

**Note:** This process can take 1-2 hours depending on network speed.

## Usage

### Starting the Server

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`

### Uploading a Song

#### Option 1: Web Interface

Open `upload.html` in your browser and use the file upload form.

#### Option 2: Command Line

Use the provided test script:

```bash
python test_upload.py path/to/your/song.mp3
```

Example output:
```
=== Analysis Results ===

Duration: 183.45 seconds

=== Audio Features ===
Tempo: 120.5 BPM
Key: C Major
Energy: 0.0234
Brightness: 1823.4

=== Top 10 Similar Songs ===
1. Artist Name - Song Title
   Similarity: 0.923
   Tempo: 118.2 BPM, Key: C Major

2. Another Artist - Another Song
   Similarity: 0.891
   Tempo: 122.0 BPM, Key: D Minor

...
```

### API Endpoint

**POST** `/analyze`

Upload an audio file to get recommendations.

**Request:**
- Content-Type: `multipart/form-data`
- Field name: `audio`
- Supported formats: MP3, WAV

**Response:**
```json
{
  "success": true,
  "duration": 183.45,
  "features": {
    "tempo": 120.5,
    "key": "C",
    "mode": "Major",
    "energy": 0.0234,
    "brightness": 1823.4
  },
  "recommendations": [
    {
      "id": "0001",
      "title": "Song Title",
      "artist": "Artist Name",
      "similarity_score": 0.923,
      "tempo": 118.2,
      "key": "C",
      "mode": "Major"
    }
  ]
}
```

## Technical Details

### Cosine Similarity Formula

The system uses cosine similarity to compare embedding vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` is the dot product of vectors A and B
- `||A||` is the L2 norm (magnitude) of vector A
- Result ranges from -1 to 1 (typically 0.5-1.0 for similar songs)

### Key Detection Algorithm

Uses the Krumhansl-Schmuckler key-finding algorithm:
1. Extract chroma features from audio
2. Compare against major/minor key profiles
3. Test all 24 possible keys (12 major + 12 minor)
4. Select key with highest correlation

## Project Structure

```
COE379 Final Project/
├── app.py                  # Flask server with analysis endpoints
├── build_database.py       # Script to build song database
├── test_upload.py         # Command-line testing tool
├── upload.html             # Web interface for uploading songs
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not in git)
├── song_database/          # Pre-processed song embeddings
│   ├── embeddings.json
│   └── audio/
└── uploads/                # Temporary upload directory
```

## Dependencies

- **Flask**: Web framework for API server
- **OpenL3**: Pre-trained audio embedding model
- **Librosa**: Audio analysis and feature extraction
- **SoundFile**: Audio file I/O
- **NumPy**: Numerical computations
- **yt-dlp**: YouTube audio downloading
- **tqdm**: Progress bars

## Limitations

- Database limited to 100 songs (configurable in `build_database.py`)
- Brute-force similarity search (O(n) complexity)
- Requires pre-downloaded audio files
- No real-time streaming support

## Future Enhancements

- Vector database (Pinecone/FAISS) for faster similarity search
- Larger song database (1000+ songs)
- Hybrid scoring with user preferences
- Web-based upload interface

## License

MIT License - Free for academic and personal use
