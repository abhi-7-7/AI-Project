# AI-Project: Audio Emotion Classifier

A modular Python project that classifies audio files by emotional category using a signal processing pipeline and a trained machine learning model.

## Overview

This project implements a full audio classification pipeline:
1. **Dataset Loading** – Loads labelled audio files from a structured data directory
2. **Signal Preprocessing** – Extracts audio features through a multi-stage pipeline (modules 1 & 2)
3. **Model Training** – Trains an `AudioModel` on extracted features
4. **Prediction** – Runs inference on a test audio file and outputs the predicted emotion class

## Project Structure

```
AI-Project/
├── data/                    # Audio dataset (labelled by emotion)
├── graphs/                  # Generated visualizations
├── module1_preprocessor/    # Stage 1 audio feature extraction
├── module2_preprocessor/    # Stage 2 audio feature refinement
├── module3_model/           # AudioModel class (train + predict)
├── dataset_loader.py        # Loads audio data from disk
└── main.py                  # Entry point: trains and runs prediction
```

## Tech Stack

- **Language:** Python
- **Libraries:** NumPy, librosa (audio processing), scikit-learn
- **Model:** Custom AudioModel with train/predict interface

## Usage

```bash
# Clone the repository
git clone https://github.com/abhi-7-7/AI-Project.git
cd AI-Project

# Install dependencies
pip install -r requirements.txt

# Add audio data to the data/ folder, organized by label:
# data/happy/, data/sad/, data/angry/, etc.

# Run the pipeline
python main.py
```

## Sample Output

```
=== RUN SUMMARY ===
Training samples: 120
Classes: {'angry', 'happy', 'neutral', 'sad'}
Test file: data/happy/sample_01.wav
Prediction: happy
```

## Author

**Aarsh Bhatnagar** – [GitHub](https://github.com/abhi-7-7)
