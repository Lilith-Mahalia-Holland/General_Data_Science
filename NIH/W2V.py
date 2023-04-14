#%load_ext tensorboard

import os
import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard.plugins.hparams import api as hp

journal_df = pd.read_csv('./NIH/data/Cleaned_NIH.csv')
df = journal_df.loc[journal_df.search_type == 'none', 'text']
df = pd.DataFrame(df).reset_index()

tokenizer = tfk.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text'])
words = pd.DataFrame({"word": tokenizer.word_index.keys(),
                      "id": tokenizer.word_index.values()})
words = words[:tokenizer.num_words].sort_values(by=["id"])

input_target = tfk.layers.Input(shape=1)
input_context = tfk.layers.Input(shape=1)
embedding = tfk.layers.Embedding(input_dim=5000 + 1,
                                 output_dim=128,
                                 input_length=1)
target_vector = tfk.layers.Flatten()(embedding(input_target))
context_vector = tfk.layers.Flatten()(embedding(input_context))
dot_product = tfk.layers.Dot(axes=1)([target_vector, context_vector])
output = tfk.layers.Dense(units=1, activation="sigmoid")(dot_product)
model = tfk.Model([input_target, input_context], output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

def skipgrams_generator(text, skip_tokenizer, window_size):
    gen = tfk.preprocessing.text.Tokenizer.texts_to_sequences_generator(skip_tokenizer, text.sample(len(text)))
    while True:
        couples = []
        labels = []
        while (len(couples)<3 and len(labels)<3):
            couples, labels = tfk.preprocessing.sequence.skipgrams(next(gen),
                                                       vocabulary_size=skip_tokenizer.num_words,
                                                       window_size=window_size,
                                                       negative_samples=1)
        x = list(map(list, zip(*couples)))
        x = [np.array(x[0]), np.array(x[1])]
        y = np.array(labels)
        yield x, y

log_dir = "./NIH/logs"
#, embeddings_metadata='metadata.tsv'
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#hparams_callback = hp.KerasCallback(log_dir, {'num_relu_units': 512,
#                                             'dropout': 0.2})
history = model.fit(skipgrams_generator(df['text'], tokenizer, 5),
                            steps_per_epoch=5000,
                            epochs=5,
                            workers=1,
                            use_multiprocessing=False)#,
                            #callbacks=[tensorboard_callback])#, hparams_callback])




#######################################################

log_dir = "./NIH/logs"
data = words.word
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    f.write('word\n')
    for word in words.word:
        f.write("{}\n".format(word))

features = tf.Variable(model.layers[2].get_weights()[0], name='features')
checkpoint = tf.train.Checkpoint(embedding=features)
checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()

embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding.metadata_path = 'metadata.tsv'

projector.visualize_embeddings(log_dir, config)

##############################################

#pd.DataFrame(model.get_weights()[0]).to_csv('./NIH/data/embedding_matrix.csv')
#words.word.to_csv('./NIH/data/words.csv')

# tensorboard --logdir "./NIH/logs" in terminal