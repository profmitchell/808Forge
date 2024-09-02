from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import soundfile as sf

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

# Load the trained VAE model
model = load_model(
    '/Users/mitchellcohen/Desktop/808Forge/models/vae_808.keras', 
    compile=False, 
    custom_objects={'sampling': sampling, 'VAELossLayer': VAELossLayer}
)

# User input for customization
distortion_level = float(input("Enter the distortion level (0-1): "))
pitch_envelope = float(input("Enter the pitch envelope level (0-1): "))
category = input("Enter the category (e.g., 'distorted', 'clean', etc.): ")

# Adjust latent space samples based on user input
latent_dim = 2  # Latent space dimension from VAE training
num_samples = 10  # Number of new samples to generate
base_samples = np.random.normal(size=(num_samples, latent_dim))

# Expand latent space to match the model's input shape
# Assuming latent_dim is part of the input shape (16), replicate samples accordingly
expanded_samples = np.tile(base_samples, (1, int(16 / latent_dim)))  # Replicate to match input shape (None, 16)

# Decode the expanded samples using the VAE
generated_features = model.predict(expanded_samples)

# Save the generated 808 sounds as .wav files
for i, features in enumerate(generated_features):
    file_path = f'/Users/mitchellcohen/Desktop/808Forge/generated_808_{category}_{i}.wav'
    sf.write(file_path, features, 44100)
    print(f'Generated 808 saved: {file_path}')
