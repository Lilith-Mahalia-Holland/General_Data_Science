import os
import pandas as pd
import numpy as np
import import_text
import tensorflow.keras as tfk
import tensorflow as tf
from tensorboard.plugins import projector


%load_ext tensorboard


data = import_text.Import_Text("finefoods.txt.gz", None)
df = data.line('ISO-8859-1', 'gzip', None, str, ':')




tokenizer = tfk.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df[5])
words = pd.DataFrame({"word": tokenizer.word_index.keys(),
                      "id": tokenizer.word_index.values()})
words = words[:tokenizer.num_words].sort_values(by=["id"])




input_target = tfk.layers.Input(shape=1)
input_context = tfk.layers.Input(shape=1)
embedding = tfk.layers.Embedding(input_dim=20000 + 1,
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



log_dir = "E:/PycharmProjects/MachineLearning/logs/finefoods/"
#, embeddings_metadata='metadata.tsv'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
history = model.fit(skipgrams_generator(df[5], tokenizer, 5),
                            steps_per_epoch=100,
                            epochs=10,
                            workers=1,
                            use_multiprocessing=False,
                            callbacks=[tensorboard_callback])
embedding_matrix = model.get_weights()[0]
embedding_matrix_df = pd.DataFrame(embedding_matrix)
history_df = pd.DataFrame(history.history)




temp = pd.DataFrame({"Unnamed: 0": [0], "word": ["UNK"], "id": [0]})
words = pd.concat([temp, words[:]]).reset_index(drop=True)
embedding_matrix_df = embedding_matrix_df.set_index(words.word.values)





with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for word in words.word[1:]:
        f.write("{}\n".format(word))

#    for unknown in range(1, 20001 - len(words.word[1:])):
#        f.write("unknown #{}\n".format(unknown))


weights = tf.Variable(model.layers[2].get_weights()[0][1:])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))


config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


%tensorboard --host localhost --logdir="E:/PycharmProjects/MachineLearning/logs/finefoods/"




# The reason the basic approach doesn't work is because the metadata is too large, look into this.
# The callback method requires the projector config and checkpoint to be properly setup
# metadata needs to be manually loaded for this to work