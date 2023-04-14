import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sme
import sklearn.manifold as sma
import tensorflow.keras as tfk
import tensorflow as tf
from tensorboard.plugins import projector
import pickle
import os
import zipfile
import time


file_regex = ["(?<=product/productId: ).*?(?=\n)",
             "(?<=review/userId: ).*?(?=\n)",
             "(?<=review/profileName: ).*?(?=\n)",
             "(?<=review/helpfulness: ).*?(?=\n)",
             "(?<=review/score: ).*?(?=\n)",
             "(?<=review/time: ).*?(?=\n)",
             "(?<=review/summary: ).*?(?=\n)",
             "(?<=review/text: ).*?(?=\n)"]
file_col_names = {0: "productId",
               1: "userId",
               2: "profileName",
               3: "helpfulness",
               4: "score",
               5: "time",
               6: "summary",
               7: "text"}


# ----------------- Data Line Import Functions -----------------
def linedata(save=False, load=False, raw_file_name=None, regex_dict=None, col_names=None, return_col=None):
    if not load:
        #pd.set_option('display.max_columns', None)
        # open file and join as single list
        raw_file = open(raw_file_name)
        raw_text = raw_file.readlines()
        raw_text = ''.join(raw_text) #.encode("utf-8")

        # seperate sections by regex filter
        # POTENTIAL BUG DUE TO INDEXING NOT INHERENTLY MATCHING, IGNORE BUG FOR NOW
        full_text = pd.DataFrame()
        #full_index = pd.DataFrame()
        for i in range(0, len(regex_dict)):
            full_text[i] = pd.DataFrame(re.findall(regex_dict[i], raw_text))
            #full_index = pd.concat([full_index, pd.DataFrame([[m.start(), m.end()] for m in re.finditer(regex_dict[i], raw_text)])], axis=1)

        # POTENTIAL PROCESSING FOR THE INDEXES, NOT SET ON WHAT THE MOST USEFUL FORM IS
        #pd.DataFrame([[m.start(), m.end()] for m in re.finditer(fileregex[0], raw_text)]).diff(axis=0).dropna(axis=1, how="all")


        # temp = full_index.to_numpy().flatten()
        # temp = np.split(temp[1:-1], len(temp[1:-1])/2)
        # temp = pd.DataFrame(temp)
        # temp = temp.diff(axis=1).dropna(axis=1, how="all")


        # g = temp.groupby(np.arange(len(temp))//8)
        #index_diff = pd.DataFrame()
        #for i in range(0, g.count().max().max()):
        #    index_diff = pd.concat([index_diff, g.nth(i).diff(axis=1)]).dropna(axis=1, how="all")


        # if na value is dataframe print
        #text_na = full_text.isnull().sum().sum()
        #index_na = full_index.isnull().sum().sum()
        #if text_na > 0 or index_na > 0:
        #    return print("na's in data set, correct data")

        # save file after processing if checked
        full_text = full_text.rename(columns=col_names)
        if save:
            full_text.to_csv("full_text.csv")
    elif load:
        full_text = pd.read_csv("../../stuff_to_clean/full_text.csv")
    return full_text[return_col]


# ----------------- Data Frame Combine -----------------
def bycsv(dfs):
    md,hd='w',True
    for df in dfs:
        df.to_csv('df_all.csv',mode=md,header=hd,index=None)
        md,hd='a',False
    #del dfs
    df_all=pd.read_csv('../../stuff_to_clean/df_all.csv', index_col=None)
    os.remove('../../stuff_to_clean/df_all.csv')
    return df_all


# ----------------- Recursive Unlist -----------------
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


# ----------------- Data Zip Import Function -----------------
def zipdata(save=False, load=False, raw_folder_name=None):
    load_base_path = "C:/Users/jdhjr/Box/COVID-19 Flattened Twitter CSVs"
    raw_folder_name = "05-2020"
    load_path = load_base_path + "/" + raw_folder_name


    load_folder_path = []
    load_file_names = []
    load_file_name = next(os.walk(load_path), (None, None, []))[2]
    for i in range(0, len(load_file_name)):
        load_file_names.append(load_file_name[i][0:-4])
        load_folder_path.append(load_path + "/" + load_file_name[i])


    # THIS SHOULD BE SEGMENTED AND COMBINED AT THE END, MAYBE CHUNKS OF AROUND 20.
    # HALF WAY THROUGH 2 SECOND FILES START TAKING 12+ SECONDS.
    # BEST CASE SHOULD TAKE AROUND 20 MINUTES, CURRENT IMPLEMENTATION TAKES AROUND 3 HOURS


    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)


    md, hd = 'w', True
    for i in range(0, 5):
        #len(load_file_name)
        if os.path.getsize(load_folder_path[i]) > 150:
            time_start = time.time()
            zf = zipfile.ZipFile(load_folder_path[i])
            temp_df = pd.read_csv(zf.open(load_file_names[i] + ".csv", "r"), usecols=["text"]).astype(str).squeeze().str.replace(emoji_pattern, "", regex=True)
            count = temp_df.str.split().str.len()
            temp_df = temp_df[~(count <= 3)]
            temp_df.replace('', np.nan, inplace=True)
            temp_df.dropna(inplace=True)
            temp_df.reset_index(drop=True)
            #temp_df.str.encode('utf-8')
            zf.close()
            temp_df.to_csv('df_all.csv', mode=md, header=hd, index=None)
            md, hd = 'a', False
            print("Reading file: {:d}/{:d}, file name: {:s}, file size: {:d}, run time: {:f}".format((i+1),
                                                                                                     (len(load_file_name)),
                                                                                                     (load_file_names[i]),
                                                                                                     (os.path.getsize(load_folder_path[i])),
                                                                                                     (time.time() - time_start)))
    return


# ----------------- Tokenizer Creation -----------------
def generate_tokenizer(save=False, load=False, full_text=None, num_words=20000, tokenizer=None):
    if not load:
        if tokenizer is None:
            tokenizer = tfk.preprocessing.text.Tokenizer(num_words=num_words)

        tokenizer.fit_on_texts(full_text)
        words = pd.DataFrame({"word": tokenizer.word_index.keys(),
                              "id": tokenizer.word_index.values()})
        words = words[:tokenizer.num_words].sort_values(by=["id"])
        if save:
            words.to_csv("words.csv")
            with open("../../stuff_to_clean/tokenizer.pickle", "wb") as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif load:
        words = pd.read_csv("../../words.csv")
        with open('../../stuff_to_clean/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    return words, tokenizer


# ----------------- Folder Walking -----------------
def folder_walk(base_path=None):
    base_path = "C:/Users/jdhjr/Box/COVID-19 Flattened Twitter CSVs"
    path = next(os.walk(base_path), (None, None, []))[1]
    mid_path = []
    for i in range(0, len(path)):
        mid_path.append(base_path + "/" + path[i])

    final_path = []
    name = []
    for i in range(0, len(mid_path)):
        file_name = next(os.walk(mid_path[i]), (None, None, []))[2]
        inner_path = []
        inner_name = []
        for j in range(0, len(file_name)):
            if file_name[j].endswith(".zip"):
                inner_path.append(mid_path[i] + "/" + os.path.splitext(file_name[j])[0] + ".zip")
                inner_name.append(os.path.splitext(file_name[j])[0] + ".csv")
        final_path.append(inner_path)
        name.append(inner_name)

    return final_path, name


# ----------------- Text cleaning -----------------
def text_clean(df):
    # squeeze, convert dtype, remove emoji, remove punctuation, remove numbers, remove special characters,
    # potentially remove non english words (this may be too resource intensive)
    return df


# ----------------- Tokenizer Out Of Memory -----------------
def generate_oom_tokenizer(folder_location=None, folder_name=None, num_words=20000, month_num=None):
    tokenizer = None
    for i in range(0, len(folder_location[month_num])):
        # replace month_num with another for loop if this needs to process al data
        zf = zipfile.ZipFile(folder_location[month_num][i])
        df = pd.read_csv(zf.open(folder_name[month_num][i]), usecols=["text"])
        df = df.squeeze()
        df = df.astype(str)

        words, tokenizer = generate_tokenizer(full_text=df, tokenizer=tokenizer, num_words=num_words)
    return words, tokenizer


# ----------------- Model Creation -----------------
def generate_model(embedding_size=128, num_words=None):
    # Make this generic so that the number of inputs and shape of can be done automatically
    input_target = tfk.layers.Input(shape=1)
    input_context = tfk.layers.Input(shape=1)

    embedding = tfk.layers.Embedding(input_dim=num_words + 1,
                               output_dim=embedding_size,
                               input_length=1)

    target_vector = tfk.layers.Flatten()(embedding(input_target))
    context_vector = tfk.layers.Flatten()(embedding(input_context))

    dot_product = tfk.layers.Dot(axes=1)([target_vector, context_vector])

    output = tfk.layers.Dense(units=1, activation="sigmoid")(dot_product)

    model = tfk.Model([input_target, input_context], output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# ----------------- Data Generators -----------------
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


# ----------------- Model Fitting -----------------
def fit_embedd_model(save=False, load=False, model=None, steps_per_epoch=None, epochs=None, generator=None,
                     full_text=None, fit_tokenizer=None, skip_window=None):
    if not load:
        history = model.fit(generator(full_text, fit_tokenizer, skip_window),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            workers=1,
                            use_multiprocessing=False)
        embedding_matrix = model.get_weights()[0]
        embedding_matrix_df = pd.DataFrame(embedding_matrix)
        history_df = pd.DataFrame(history.history)
        if save:
            embedding_matrix_df.to_csv("embedding_matrix.csv")
            history_df.to_csv("history.csv")
    elif load:
        embedding_matrix_df = pd.read_csv("../../stuff_to_clean/embedding_matrix.csv")
        history = pd.read_csv("../../stuff_to_clean/history.csv")
    return embedding_matrix_df, history


# ----------------- T-sne Plot -----------------
def tsne_plot(dimension=3, perplexity=50, embedding_matrix=None, tsne_length=500, words_df=None):
    tsne = sma.TSNE(n_components=dimension,
                    perplexity=perplexity,
                    learning_rate="auto",
                    init="pca").fit_transform(embedding_matrix[:tsne_length])
    tsne = tsne.transpose()

    if dimension == 3:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(tsne[0], tsne[1], tsne[2])
        for i, label in enumerate(words_df.word[:tsne_length]):
           ax.text(s=label, x=tsne[0][i], y=tsne[1][i], z=tsne[2][i])
    elif dimension == 2:
        plt.scatter(tsne[0], tsne[1])
        for i, label in enumerate(words_df.word[:tsne_length]):
            plt.annotate(label, (tsne[0][i], tsne[1][i]))


# ----------------- Similar Words -----------------
def find_similar_words(word, embedding_matrix, dataframe, n=10):
    index = dataframe.index[dataframe.word == word]
    similarities = pd.DataFrame(sme.pairwise.cosine_similarity(embedding_matrix,
                                                               embedding_matrix[index.item():(index.item() + 1)]),
                                                               columns=["similarity"])
    similarities["word"] = dataframe.word[:len(similarities)]
    similarities = similarities.sort_values(by="similarity", ascending=False)
    return similarities.head(n)


# ----------------- Plot History -----------------
def plot_history(history):
    plt.figure(figsize=(2, 1))
    plt.subplot(2, 1, 1)
    plt.plot(range(0, 10), history["loss"])
    plt.subplot(2, 1, 2)
    plt.plot(range(0, 10), history["accuracy"])
    plt.show()


# ----------------- T-sne Generator -----------------
def tsne_perp_gen(dim=None, tsne_length=None, perp_lim=None, save=False, load=False, embedding_matrix=None):
    if not load:
        tsne_array = np.empty((perp_lim,dim,tsne_length))
        for i in range(1,perp_lim+1):
            tsne = sma.TSNE(n_components=dim,
                            perplexity=i,
                            learning_rate="auto",
                            init="pca").fit_transform(embedding_matrix[:tsne_length])
            tsne = tsne.transpose()
            tsne = np.expand_dims(tsne, 0)
            tsne_array[i-1] = tsne
        if save:
            np.save("tsne_perplexity", tsne_array)
    if load:
        tsne_array = np.load("../../stuff_to_clean/tsne_perplexity.npy")
    return tsne_array


# ----------------- Projector -----------------
#def projector():
#    log_dir = "/logs/embedding"
#    if not os.path.exists(log_dir):
#        os.makedirs(log_dir)
#
#    with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
#        for subword in encoder.subwords:


# ----------------- Main Code -----------------
#reviews = linedata(load=True, return_col="text")
reviews = zipdata()
words, tokenizer = generate_tokenizer(full_text=reviews)
model = generate_model(num_words=words.shape[0])
print(model.summary())
#embedding_matrix_df, history = fit_embedd_model(save=False, model=model, steps_per_epoch=10000, epochs=10,
#                                                generator=skipgrams_generator, full_text=reviews,
#                                                fit_tokenizer=tokenizer, skip_window=5)
#embedding_matrix_df, history = fit_embedd_model(load=True)
embedding_matrix_df, history = fit_embedd_model(model=model, steps_per_epoch=10000, epochs=10,
                                                generator=skipgrams_generator, full_text=reviews,
                                                fit_tokenizer=tokenizer, skip_window=5)


#embedding_matrix_df = embedding_matrix_df.drop("Unnamed: 0", 1)
temp = pd.DataFrame({"Unnamed: 0": [0], "word": ["UNK"], "id": [0]})
words = pd.concat([temp, words[:]]).reset_index(drop=True)
embedding_matrix_df = embedding_matrix_df.set_index(words.word.values)


#find_similar_words("love", embedding_matrix_df, words)
#tsne_plot(dimension=2, embedding_matrix=embedding_matrix_df, words_df=words)


# ADD STOPWORD CLEANING, ADD PROJECTOR, LOOK INTO STOPWORD REMOVAL AND REASON


#for i in range(0,50):
#    fig = plt.figure(figsize=(12, 12))
#    ax = fig.add_subplot(projection="3d")
#    ax.scatter(tsne[i,0], tsne[i,1], tsne[i,2])
#    for j, label in enumerate(words.word[:tsne.shape[2]]):
#        ax.text(s=label, x=tsne[i][0][j], y=tsne[i][1][i], z=tsne[i][2][i])


