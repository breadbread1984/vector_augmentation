#!/usr/bin/python3

from absl import app, flags
import tensorflow as tf
from models import VAE
from create_dataset import parse_function

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to tfrecord')
  flags.DEFINE_integer('batch_size', default = 128, help = 'batch size')
  flags.DEFINE_float('lr', default = 1e-3, help = 'learning rate')

def main(unused_argv):
  vae = VAE()
  trainset = tf.data.TFRecordDataset(FLAGS.dataset).map(parse_function).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size)batch(FLAGS.batch_size)
  vae.compile(optimizer = tf.keras.optimizers.Adam(FLAGS.lr), loss = lambda x, sample: -sample.log_prob(x))
  vae.fit(trainset, epochs = 15)
  vae.save_weights('vae.keras')

if __name__ == "__main__":
  add_options()
  app.run(main)

