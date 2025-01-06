#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 09:30:38 2024

@author: boscolo
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled = matmul_qk / tf.math.sqrt(dk)  # scaled by the square root of the dimensionality of the key
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled += (mask * -1e9)  # -1e9 ~ (-INFINITY) => wherever mask is set, make its scaled value close to -INF
    # softmax to get attention weights
    attention_weights = tf.nn.softmax(scaled, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        # Dense layers for query, key, and value
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        # Apply dense layers to query, key, and value
        query = self.query_dense(query)  # (batch_size, seq_len, d_model)
        key = self.key_dense(key)  # (batch_size, seq_len, d_model)
        value = self.value_dense(value)  # (batch_size, seq_len, d_model)

        # Split heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_q, depth)

        # Perform scaled dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        outputs = self.dense(concat_attention)  # (batch_size, seq_len, d_model)
        return outputs, attention_weights


class Encoder(tf.keras.Model):

    def __init__(self, IMAGE_H, IMAGE_W, latent_dim, encoder_filters, attention_layer):
        super(Encoder, self).__init__()
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.attention_layer = attention_layer

        self.conv_layers = []
        for filters in encoder_filters:
            self.conv_layers.append(layers.Conv2D(filters, kernel_size=3, activation='relu', padding='same'))
            self.conv_layers.append(layers.MaxPooling2D(pool_size=2))

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(16 * (self.IMAGE_H  // 4) * (self.IMAGE_W  // 4), activation='relu')
        self.z_mean = layers.Dense(self.latent_dim)
        self.z_log_var = layers.Dense(self.latent_dim)

    def call(self, inputs, error_labels):
        x = inputs # (batch, 48,48,3)
        #breakpoint()
        for layer in self.conv_layers:
            x = layer(x) 
        ### dim after 2d conv [batch, 12,12,32]

         # Reshape for attention
        batch_size, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, height * width, channels))  # Shape: (batch_size, 144, 32)

        attention_output, attention_weights = self.attention_layer(x, x, x)
        x = layers.Add()([x, attention_output])  # Shape: (batch_size, 121, 32)

        x_flat = self.flatten(x) # (batch, 3872)
        x = layers.Concatenate()([x_flat, error_labels]) # (batch, 3872+label_dim)
        
        x = self.dense(x) #(batch,7744)
        z_mean = self.z_mean(x)# (batch, 10)
        z_log_var = self.z_log_var(x)

        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.latent_dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return z_mean, z_log_var, z


class Decoder(tf.keras.Model):

    def __init__(self, IMAGE_H, IMAGE_W, latent_dim, decoder_filters):
        super(Decoder, self).__init__()
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.latent_dim = latent_dim
        #self.num_error_categories = num_error_categories
        self.decoder_filters = decoder_filters

        self.dense = layers.Dense(16 * (self.IMAGE_H // 4)*(self.IMAGE_W // 4), activation='relu')
        self.reshape = layers.Reshape((self.IMAGE_H // 4, self.IMAGE_W // 4, 16))

        self.conv_layers = []
        for filters in decoder_filters:
            self.conv_layers.append(layers.Conv2D(filters, kernel_size=3, activation='relu', padding='same'))
            self.conv_layers.append(layers.UpSampling2D(size=2))  # Use UpSampling2D for 2D images

        # Final Conv2D layer to produce the RGB output
        # Input: [batch, h, w, channels]
        # Output: [batch, h, w, 3] (RGB image)
        self.final_conv = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')

        #self.flatten = layers.Flatten()
        #self.output_layer = layers.Dense(self.input_dim * self.input_dim * 3, activation='sigmoid')  # For reshaping to RGB image size

    def call(self, latent_inputs, error_labels):
        x = layers.Concatenate()([latent_inputs, error_labels]) # (batch, latentdim+label_dim)
        x = self.dense(x) #(batch,2304)
        x = self.reshape(x) #(batch,12,12,16)
       
        for layer in self.conv_layers:
            x = layer(x)
            
        x = self.final_conv(x) ## (batch,48,48,3)
        #x = self.flatten(x)
        return x #self.output_layer(x)


class ConditionalVariationalAutoencoder(tf.keras.Model):
    def __init__(self, 
                 IMAGE_H,
                 IMAGE_W ,
                 label_dim,
                 latent_dim,
                 learning_rate,
                 encoder_filters=(32, 16),
                 decoder_filters=(16, 32)):
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate

        self.attention_layer = MultiHeadAttention(d_model=encoder_filters[-1], num_heads=2)
        self.encoder = Encoder(IMAGE_H, IMAGE_W, latent_dim, encoder_filters, self.attention_layer)
        self.decoder = Decoder(IMAGE_H, IMAGE_W, latent_dim, decoder_filters)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    # def call(self, inputs, error_labels):
    #     z_mean, z_log_var, z = self.encoder(inputs, error_labels)
    #     reconstructed = self.decoder(z, error_labels)
    #     return reconstructed, z_mean, z_log_var
    
    
    def load_checkpoint(self, checkpoint_dir):
      # Create a checkpoint object
      self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
      
      # Create a checkpoint manager
      self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=1)
      
      # Restore from the latest checkpoint if available
      if self.manager.latest_checkpoint:
          self.checkpoint.restore(self.manager.latest_checkpoint)
          print(f"Restored from checkpoint: {self.manager.latest_checkpoint}")
      else:
          print("Starting training session without checkpoint")
    
    
    def fit(self, 
            data,
            data_labels, 
            data_val,
            data_labels_val,
            epochs=100,
            batch_size=16,
            patience=15,
            early_stopping_interval=10,
            checkpoint_dir='checkpoints'):
        
        try:
            
            self.checkpoint_dir = checkpoint_dir # Define the checkpoint directory
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=1)
            
            
            if self.manager.latest_checkpoint:
                self.checkpoint.restore(self.manager.latest_checkpoint)
                print(f"Restored from checkpoint: {self.manager.latest_checkpoint}")
            else:
                print("Starting training session without checkpoint")
            
            
            # Resume training if checkpoint exists --> old version 
            # if os.path.exists(checkpoint_dir):
            #     print(f"Resuming training from checkpoint: {checkpoint_dir}")
            #     self.load_weights(checkpoint_dir)
            # else:
            #     print("Starting training session without checkpoint")
    
            num_samples = data.shape[0]
            best_loss = np.inf
            wait_val = 0
            wait_train = 0
            min_learning_rate = 1e-5
            initial_lr = self.learning_rate
    
            for epoch in range(epochs):
                epoch_loss = 0
                if epoch > 0 and epoch % 20 == 0:
                # Adjust learning rate linearly
                    new_lr = max(min_learning_rate, initial_lr * (1 - epoch / epochs))
                    self.optimizer.learning_rate.assign(new_lr)
                    print(f"Epoch {epoch + 1}/{epochs}, Learning rate: {new_lr:.8f}")
    
                # Training loop
                for i in range(0, num_samples, batch_size):
                    batch_data = data[i:i + batch_size]
                    batch_labels = data_labels[i:i + batch_size]
    
                    with tf.GradientTape() as tape:
                        z_mean, z_log_var, z = self.encoder(batch_data, batch_labels)
                        batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
                        reconstructed = self.decoder(z, batch_labels_tensor)
    
                        # Compute losses
                        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch_data, reconstructed))
                        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                        vae_loss = reconstruction_loss + kl_loss
    
                    gradients = tape.gradient(vae_loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                    epoch_loss += vae_loss.numpy()
    
                avg_loss = epoch_loss / (num_samples // batch_size)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
                # # Save checkpoint if best loss is improved
                # if avg_loss < best_loss:
                #     best_loss = avg_loss
                #     self.manager.save()
                #     #self.save_weights(checkpoint_dir)
                #     wait_train = 0  # Reset wait counter when improvement is observed
                # else:
                #     wait_train += 1
                #     #breakpoint()
                #     if wait_train >= patience:
                #         print("Early stopping triggered due to no improvement in training loss.")
                #         break
    
                # Validation loss check and early stop implementation 
                if data_val is not None and (epoch + 1) % early_stopping_interval == 0:
                    val_loss = self.evaluate(data_val, data_labels_val, batch_size)
                    print(f"Validation Loss: {val_loss:.4f}")
                    if val_loss < best_loss:
                        self.manager.save()
                        wait_val = 0
                    else:
                        wait_val += 1
                        if wait_val >= patience:
                            print("Early stopping triggered due to validation loss.")
                            break
        
        except KeyboardInterrupt:
                print("Keyboard interrupt received. Saving checkpoint ...")
                # Save model weights if there's an improvement
                #checkpoint_path = os.path.join(checkpoint_dir, f'vae_1D_best_weights.h5')
                self.manager.save()
                #self.save_weights(checkpoint_dir)
                print(f'Model weights saved at: {checkpoint_dir}')
        finally:
                # TBD
                print("Training process ended.")
                #breakpoint()
        

    def evaluate(self, data, error_labels, batch_size):
        num_samples = data.shape[0]
        total_loss = 0

        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = error_labels[i:i + batch_size]

            z_mean, z_log_var, z = self.encoder(batch_data, batch_labels)
            batch_labels_tensor = tf.convert_to_tensor(batch_labels, dtype=tf.float32)
            reconstructed = self.decoder(z, batch_labels_tensor)

            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch_data, reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss += (reconstruction_loss + kl_loss).numpy()

        return total_loss / (num_samples // batch_size)
    
    def encode(self, data, labels):
        """Encode data points into latent space with labels."""
        z_mean, z_log_var, z = self.encoder(data, labels) 
        return z_mean, z_log_var,z

    def decode(self, z, labels):
        """Reconstruct data points from latent space with labels."""
        return self.decoder(z, labels) 

