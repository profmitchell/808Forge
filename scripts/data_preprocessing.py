import os
import librosa
import numpy as np
import pandas as pd

# Define the path to your samples folder
samples_folder = '/Users/mitchellcohen/Desktop/808Forge/samples'
output_file = 'scripts/preprocessed_data.parquet'

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=44100)

        # Dynamically adjust n_fft and hop_length based on the length of the audio signal
        n_fft = min(256, len(y))  # Slightly increase n_fft but keep it within the length of the audio
        hop_length = max(1, n_fft // 4)  # Ensure hop_length is at least 1
        n_mels = 20  # Reduce the number of Mel bands to avoid empty filters

        # Extract audio features that are suitable for short audio clips
        features = {
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)),
            'rms': np.mean(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)),
            'mfccs': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13, n_mels=n_mels), axis=1).tolist(),
        }

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare the dataset
data = []
for filename in os.listdir(samples_folder):
    if filename.endswith('.aif') or filename.endswith('.wav'):  # Ensure it captures all file types
        file_path = os.path.join(samples_folder, filename)
        features = extract_features(file_path)
        if features:
            # Add the filename and extracted features to the dataset
            features['filename'] = filename
            data.append(features)

# Convert the dataset to a DataFrame and save as Parquet
df = pd.DataFrame(data)
df.to_parquet(output_file, index=False)
print(f"Data preprocessing completed and saved to {output_file}")
