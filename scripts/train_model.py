import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# Load the preprocessed data
df = pd.read_parquet('/Users/mitchellcohen/Desktop/808Forge/scripts/preprocessed_data.parquet')

# Expand the MFCCs into separate columns
mfccs_df = pd.DataFrame(df['mfccs'].tolist(), index=df.index)
df = pd.concat([df.drop(columns=['mfccs', 'filename']), mfccs_df], axis=1)

# Convert all column names to strings
df.columns = df.columns.astype(str)

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(df)

# Define the VAE model architecture
input_dim = X.shape[1]  # Number of features
latent_dim = 2  # Dimensionality of the latent space

inputs = Input(shape=(input_dim,))
h = Dense(64, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Custom sampling function for latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h = Dense(64, activation='relu')
decoder_mean = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom Keras layer for VAE loss
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        reconstruction_loss = MeanSquaredError()(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        total_loss = K.mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return x_decoded_mean

# Apply the custom loss layer
outputs = VAELossLayer()([inputs, x_decoded_mean, z_mean, z_log_var])

# Define the full VAE model
vae = Model(inputs, outputs)

# Compile the model
vae.compile(optimizer=Adam())

# Train the VAE
vae.fit(X, X, epochs=100, batch_size=32, validation_split=0.2)

# Save the trained model
vae.save('/Users/mitchellcohen/Desktop/808Forge/models/vae_808.keras')
