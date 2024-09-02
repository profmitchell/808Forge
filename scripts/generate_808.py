from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import soundfile as sf
import librosa

# Register the custom sampling function
@register_keras_serializable()
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Register the custom loss layer
@register_keras_serializable()
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return reconstruction_loss + kl_loss

    def call(self, inputs):
        inputs, outputs, z_mean, z_log_var = inputs
        loss = self.vae_loss(inputs, outputs, z_mean, z_log_var)
        self.add_loss(loss)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[1]

# Load preprocessed data
df = pd.read_parquet('/Users/mitchellcohen/Desktop/808Forge/scripts/preprocessed_data.parquet')

# Convert all column names to strings
df.columns = df.columns.astype(str)

# Identify columns with list or array-like structures
list_columns = df.select_dtypes(include=[list]).columns

# Expand list columns into separate columns
for col in list_columns:
    expanded_cols = pd.DataFrame(df[col].tolist(), index=df.index)
    expanded_cols.columns = [f"{col}_{i}" for i in range(expanded_cols.shape[1])]
    df = pd.concat([df.drop(columns=[col]), expanded_cols], axis=1)

# Select only scalar columns for scaling
scalar_columns = df.select_dtypes(include=[np.number]).columns
X_scaled = MinMaxScaler().fit_transform(df[scalar_columns])

# Load your trained model
model = load_model(
    '/Users/mitchellcohen/Desktop/808Forge/models/vae_808.keras',
    custom_objects={'sampling': sampling, 'VAELossLayer': VAELossLayer}
)

# Get user input for customization
distortion_level = float(input("Enter the distortion level (0-1): "))
pitch_envelope_level = float(input("Enter the pitch envelope level (0-1): "))
category = input("Enter the category (e.g., 'distorted', 'clean', etc.): ")

# Generate initial sample (10 random samples)
input_dim = model.input_shape[-1]  # Fetch the input dimension from the model

# Generate random samples with the correct shape
random_samples = np.random.normal(size=(10, input_dim))

# Apply user inputs to the generated samples
customized_samples = random_samples * distortion_level + pitch_envelope_level

# Introduce slight variations to each sample
variation_factor = 0.1  # Change this value to control the amount of variation
for i in range(customized_samples.shape[0]):
    customized_samples[i] += np.random.normal(scale=variation_factor, size=customized_samples[i].shape)

# Predict and generate audio features using the trained model
generated_features = model.predict(customized_samples)

# Ensure minimum length of 1 second for each generated audio
min_length = 44100  # 1 second at 44.1 kHz sample rate

# Target pitch frequency for D0 (approximately 36.71 Hz)
target_pitch_freq = 36.71

# Normalize and extend generated features to at least 1 second
for i, features in enumerate(generated_features):
    # Normalize between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    # Extend the sound to 1 second by padding or repeating
    if len(features) < min_length:
        repeat_count = int(np.ceil(min_length / len(features)))
        extended_features = np.tile(features, repeat_count)[:min_length]
    else:
        extended_features = features

    # Estimate the current pitch using librosa's piptrack
    pitches, magnitudes = librosa.core.piptrack(y=extended_features, sr=44100)
    current_pitch_freq = np.max(pitches)

    # Calculate pitch shift steps to target D0
    pitch_shift_steps = librosa.core.hz_to_midi(target_pitch_freq) - librosa.core.hz_to_midi(current_pitch_freq)
    extended_features = librosa.effects.pitch_shift(extended_features, sr=44100, n_steps=pitch_shift_steps)

    # Save the extended audio
    file_path = f"/Users/mitchellcohen/Desktop/808Forge/generated_808_{category}_{i}.wav"
    sf.write(file_path, extended_features, 44100)
    print(f"Generated 808 saved: {file_path}")
