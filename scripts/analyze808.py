import librosa
import numpy as np
import os

# Function to analyze the 808 sample
def analyze_808(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=44100)

    # Extract features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)

    # Return the features for further comparison
    return {
        'file_path': file_path,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'zero_crossing_rate': zero_crossing_rate,
        'rms': rms,
        'mfccs': mfccs
    }

# Folder containing your 808 samples
folder_path = '/Users/mitchellcohen/Desktop/808Forge/LowSamples'
output_file_path = '/Users/mitchellcohen/Desktop/808Forge/808_analysis_results.txt'

# Prepare a list to collect all results
analysis_results = []

# Iterate through all .aif files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.aif'):
        file_path = os.path.join(folder_path, filename)
        features = analyze_808(file_path)
        analysis_results.append(features)

# Write the analysis results to a text file
with open(output_file_path, 'w') as f:
    for result in analysis_results:
        f.write(f"Analysis of {result['file_path']}:\n")
        f.write(f"Spectral Centroid: {result['spectral_centroid']}\n")
        f.write(f"Spectral Bandwidth: {result['spectral_bandwidth']}\n")
        f.write(f"Zero Crossing Rate: {result['zero_crossing_rate']}\n")
        f.write(f"RMS Energy: {result['rms']}\n")
        f.write(f"MFCCs: {result['mfccs'].tolist()}\n\n")

print(f"Analysis complete. Results saved to {output_file_path}")
