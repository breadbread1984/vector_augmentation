#!/usr/bin/python3

import tensorflow as tf
import tensorflow_probability as tfp

def Encoder(code_size = 4):
  inputs = tf.keras.Input((12,)) # inputs.shape = (batch, 12)
  results = tf.keras.layers.Dense(128, activation = tf.keras.activations.gelu)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  results = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(code_size))(results)
  prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(code_size), scale = 1), reinterpreted_batch_ndims = 1)
  results = tfp.layers.MultivariateNormalTriL(code_size, activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = 1.0))(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

def Decoder(code_size = 4):
  inputs = tf.keras.Input((code_size,)) # inputs.shape = (batch, code_size)
  results = tf.keras.layers.Dense(128, activation = tf.keras.activations.gelu)(inputs)
  results = tf.keras.layers.BatchNormalization()(results)
  results = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(12))(results)
  results = tfp.layers.MultivariateNormalTriL(12)(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

class VAE(tf.keras.Model):
  def __init__(self, code_size = 4):
    super(VAE, self).__init__()
    self.encoder = Encoder(code_size)
    self.decoder = Decoder(code_size)
  def call(self, inputs):
    codes = self.encoder(inputs)
    samples = self.decoder(codes)
    return samples

if __name__ == "__main__":
  vae = VAE()
  inputs = tf.random.normal(shape = (8, 12))
  outputs = vae(inputs)
  print(outputs.shape)
