# Music Recommendation System Using Audio Embeddings

A machine learning-based music recommendation system that analyzes audio files using OpenL3 embeddings and Librosa feature extraction to find similar songs.

## Overview

This project implements a content-based music recommendation system that uses deep learning and signal processing techniques to analyze audio similarity. The system extracts acoustic features from uploaded songs and compares them against a database of pre-processed tracks to find the most similar matches.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Input Audio File                            │
│                         (MP3, WAV format)                            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Audio Preprocessing        │
              │   - Load audio waveform      │
              │   - Convert to mono          │
              │   - Resample if needed       │
              └──────────┬───────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────────┐         ┌────────────────────┐
│  OpenL3 Embedding  │         │ Librosa Features   │
│  Extraction        │         │ Extraction         │
├────────────────────┤         ├────────────────────┤
│ • 512D vector      │         │ • Tempo (BPM)      │
│ • Timbre           │         │ • Key & Mode       │
│ • Rhythm           │         │ • Energy (RMS)     │
│ • Harmonic content │         │ • Brightness       │
└────────┬───────────┘         └────────┬───────────┘
         │                               │
         │      70% weight               │  30% weight
         └───────────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │   Hybrid Similarity Score    │
              │                              │
              │   Final Score = 0.70 × cos   │
              │   similarity(OpenL3) + 0.30  │
              │   × Librosa feature match    │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │  Compare with Database       │
              │  (Brute-force O(n) search)   │
              └──────────┬───────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │   Top 10 Recommendations     │
              │   Sorted by similarity score │
              └──────────────────────────────┘
```

### Pipeline Components

#### 1. OpenL3 Audio Embeddings
- Pre-trained convolutional neural network trained on Google's AudioSet
- Converts audio waveforms into 512-dimensional embedding vectors
- Captures semantic audio features (timbre, rhythm, harmonic content)
- Similarity measured using cosine similarity between embedding vectors

#### 2. Librosa Feature Extraction
- Tempo detection using beat tracking algorithms
- Key and mode detection using chroma features and Krumhansl-Schmuckler profiles
- Energy analysis using RMS (Root Mean Square)
- Spectral brightness using spectral centroid

#### 3. Hybrid Scoring Formula
The final similarity score combines both approaches:
- **70% OpenL3 embeddings**: Captures complex acoustic patterns and perceptual similarity
- **30% Librosa features**: Ensures compatibility in tempo, key, energy, and brightness

### Recommendation Process
1. User uploads an audio file (MP3 or WAV)
2. System extracts OpenL3 embedding (512D vector) and Librosa features
3. Hybrid similarity score computed between upload and all database songs
4. Top 10 most similar songs returned with similarity scores (0.0-1.0 range)

## Installation

### Prerequisites
- **Python 3.10 or 3.11** (required - Python 3.12+ not yet supported by OpenL3)
- pip (included with Python)

### Setup

1. Clone or download this project

2. **Install Python 3.11** if you don't have it:
   - Download from https://www.python.org/downloads/release/python-31110/
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version` or `py -3.11 --version`

3. Create a virtual environment:

**Windows (PowerShell):**
```bash
# If you have Python 3.11 as your default Python:
python -m venv music-rec
.\music-rec\Scripts\Activate.ps1

# If you have multiple Python versions installed:
py -3.11 -m venv music-rec
.\music-rec\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
# If you have Python 3.11 as your default Python:
python -m venv music-rec
.\music-rec\Scripts\activate.bat

# If you have multiple Python versions installed:
py -3.11 -m venv music-rec
.\music-rec\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3.11 -m venv music-rec
source music-rec/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up Last.fm API key:
   - Get a free API key at https://www.last.fm/api/account/create
   - Copy `.env.example` to `.env`
   - Add your API key to `.env`

## Building the Song Database

Before running the system, you need to build a database of songs:

```bash
python build_database.py
```

This script will:
- Fetch the top 999 songs from Last.fm
- Download audio from YouTube
- Extract OpenL3 embeddings and Librosa features
- Save everything to `song_database/embeddings.json`

**Note:** This process can take 1-2 hours depending on network speed.

## Usage

### Step 1: Start the Flask Server

In your terminal with the virtual environment activated:

```bash
python app.py
```

The Flask server will start on `http://localhost:5000`. Keep this terminal window open.

### Step 2: Test with Your Music

1. **Add your song file** to the project directory (any MP3 or WAV file)
   - Example: Place `mysong.mp3` in `C:\Users\jasom\Documents\coe379L\project3\`

2. **Open a new terminal** (keep the server running in the first one)

3. **Activate the virtual environment** in the new terminal:
   ```bash
   .\music-rec\Scripts\Activate.ps1
   ```

4. **Run the test script** with your song:
   ```bash
   python test_upload.py mysong.mp3
   ```

### Example Output

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

- **Python 3.11**: Main development language for preprocessing, feature extraction, and similarity computation
- **Flask**: Web framework providing REST API endpoints for audio upload and analysis
- **Librosa**: Audio analysis library for extracting musical descriptors (tempo, key) using chroma features and beat tracking algorithms
- **OpenL3**: Pre-trained deep-learning model (trained on Google's AudioSet) that generates 512-dimensional embedding vectors capturing timbre, rhythm, and harmonic characteristics
- **NumPy**: Vector manipulation, normalization, and cosine similarity calculations between embeddings
- **SoundFile**: Audio file I/O for loading WAV and MP3 files
- **requests**: HTTP library for testing the API with command-line scripts
- **yt-dlp**: YouTube audio downloading for building the song database from Last.fm track listings