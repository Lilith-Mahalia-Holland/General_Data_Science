import os
import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector
import codecs

def create_embeddings(sess, log_dir, embedding_file='', tensor_name='embedding'):
  """ Add the embeddings to input TensorFlow session and writes a metadata_file containing the words in the vocabulary
  :param sess: TF session
  :param log_dir: destination directory for the model and metadata (the one to which TensorBoard points)
  :param embedding_file: embedding file
  :param tensor_name: tensor name
  :return:
  """
  embedding = None
  embedding_dimensions = 0
  vocab_size = 0
  # write labels
  with open(os.path.join(log_dir, tensor_name + '_' + 'metadata.tsv'), 'w') as metadata_file:
    print('---', embedding_file)
    with codecs.open(embedding_file, 'r', encoding='ISO-8859-1') as inputfile:

      for i, line in enumerate(inputfile):

        line = line.rstrip()
        values = line.split()

        # the first line is always the header based on what we produce in the embeddings_knn.py
        if i == 0:
          vocab_size = int(values[0])
          embedding_dimensions = int(values[1])
          embedding = np.empty((vocab_size, embedding_dimensions), dtype=np.float32)
        else:
          # accounts for the case of words with spaces
          word = ' '.join(values[0:len(values) - embedding_dimensions]).strip()
          coefs = np.asarray(values[-embedding_dimensions:], dtype='float32')
          embedding[i - 1] = coefs
          metadata_file.write(word + '\n')

  #X = tf.Variable([0.0], name=tensor_name)
  #place = tf.keras.Input(shape=embedding.shape)
  X = tf.Variable([0.0], shape=tf.TensorShape(None))
  set_x = X.assign(embedding.shape)
  #set_x = tf.compat.v1.assign(X, list(embedding.shape))

  #place = tf.placeholder(tf.float32, shape=embedding.shape)
  #set_x = tf.assign(X, place, validate_shape=False)

  #sess.run(set_x, feed_dict={place: embedding})


def add_multiple_embeddings(log_dir, embed_f):
  """ Creates the files necessary for the multiple embeddings
  :param log_dir: destination directory for the model and metadata (the one to which TensorBoard points)
  :param file_list: list of embeddings files
  :param name_list: names of the embeddings files
  :return:
  """
  # setup a TensorFlow session
  #tf.reset_default_graph()
  #sess = tf.InteractiveSession()
  #sess.run(tf.global_variables_initializer())
  config = projector.ProjectorConfig()
  sess=None
  #file = embed_dir + '/' + fn
  tensor_name = "embed" # file.split("/")[-1]

  print('creating the embedding with the name ' + tensor_name)
  create_embeddings(sess, log_dir, embedding_file=embed_f,
                    tensor_name=tensor_name)
  # create a TensorFlow summary writer
  #summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
  summary_writer = tf.summary.create_file_writer(log_dir)

  embedding_conf = config.embeddings.add()
  embedding_conf.tensor_name = tensor_name + ':0'
  embedding_conf.metadata_path = os.path.join(tensor_name + '_' + 'metadata.tsv')
  projector.visualize_embeddings(log_dir, config)

  # save the model
  #saver = tf.train.Saver()
  #saver.save(sess, os.path.join(log_dir, tensor_name + '_' + "model.ckpt"))

  print('finished successfully!')

add_multiple_embeddings('logdir', 'temp.txt')



