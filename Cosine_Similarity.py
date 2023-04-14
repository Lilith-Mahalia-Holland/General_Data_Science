import pandas as pd
import numpy as np
import sklearn.metrics as sme
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import seaborn as sn

def find_similar_words(word, embedding_matrix, dataframe, n=10):
    index = dataframe.index[dataframe.word == word]
    similarities = pd.DataFrame(sme.pairwise.cosine_similarity(embedding_matrix,
                                                               embedding_matrix[index.item():(index.item() + 1)]),
                                                               columns=["similarity"])
    similarities["word"] = dataframe.word[:len(similarities)]
    similarities = similarities.sort_values(by="similarity", ascending=False)
    return similarities.head(n)

embedding_matrix = pd.read_csv('./NIH/data/embedding_matrix.csv')
words = pd.read_csv('./NIH/data/words.csv')
journal_df = pd.read_csv('./NIH/data/Cleaned_NIH.csv')


target_words = ['effectiveness', 'harassment', 'usefulness', 'value', 'evaluation']
target_df = []

for target in target_words:
    target_df.append(' '.join(find_similar_words(target, embedding_matrix, words).word.values))

# I think this could be simplified if I made the data types of the df correct
compressed_journal_df = (journal_df
                         .loc[journal_df.search_type == 'none', :]
                         .groupby(['path', 'paper_number'])['text']
                         .apply(lambda x: ' '.join(x))
                         .reset_index())
# tokenize each document individually to create a tokenized df

vector_df = pd.concat([pd.Series(target_df), compressed_journal_df['text']]).reset_index(drop=True).to_list()

count_vectorizer = CountVectorizer()
vector_matrix = count_vectorizer.fit_transform(vector_df)
vector_matrix

tokens = count_vectorizer.get_feature_names_out()
len(tokens)

vector_matrix.toarray()

def create_dataframe(matrix, tokens):
    doc_name = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_name, columns=tokens)
    return(df)

create_dataframe(vector_matrix.toarray(), tokens)

cosine_similarity_matrix = cosine_similarity(vector_matrix)


final_df = create_dataframe(cosine_similarity_matrix, target_words + compressed_journal_df.index.to_list())

doc_names = [f'doc_{i+1}' for i, _ in enumerate(compressed_journal_df['text'])]
final_df = final_df[target_words].set_axis(target_words + doc_names, axis=0)

final_df[:] = np.tril(final_df.values, k=-1)

sn.heatmap(final_df[:][len(target_words):], yticklabels=True, vmax=0.10)
plt.xticks(rotation=0)
plt.show()















import time

import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns





df_list = []
for target in target_words:
    temp_df = embedding_matrix[embedding_matrix.index.isin(find_similar_words(target, embedding_matrix, words, n=20).word.index.to_list())]
    temp_df.loc[:, 'group'] = target
    df_list.append(temp_df)

target_df = pd.concat(df_list)
target_df = target_df.rename(columns={target_df.columns[0]: "words"})

words.word[target_df.index]
target_df.words = words.word[target_df.index]
target_df = target_df.reset_index()

target_df_subset = target_df.loc[:, ~target_df.columns.isin(['words', 'index', 'group'])]

for perp in range(30, 105, 5):
    for iter in range(250, 1050, 50):
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=iter)
        tsne_results = tsne.fit_transform(target_df_subset)

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        df_subset = pd.DataFrame()
        df_subset['x_val'] = tsne_results[:, 0]
        df_subset['y_val'] = tsne_results[:, 1]
        df_subset['group'] = target_df.group

        sns.scatterplot(
            x="x_val", y="y_val",
            hue="group",
            palette=sns.color_palette("hls", len(df_subset.groupby('group').count())),
            data=df_subset,
            legend="full",
            alpha=1
        )

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(target_df_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df_subset = pd.DataFrame()
df_subset['x_val'] = tsne_results[:, 0]
df_subset['y_val'] = tsne_results[:, 1]
df_subset['group'] = target_df.group

# THIS IS WHAT NEEDS TO BE WORKED ON, MAYBE MAKE TEXT BACKGROUND MATCH HUE
sns.scatterplot(
    x="x_val", y="y_val",
    hue="group",
    palette=sns.color_palette("hls", len(df_subset.groupby('group').count())),
    data=df_subset,
    legend="full",
    alpha=1,
    s=200
)
sns.set_style("dark")

plt.title('T-SNE target word groups')
plt.xlabel('x')
plt.ylabel('y')

for index, value in enumerate(target_df.words):
    plt.text(x=df_subset.x_val[index]+0.3, y=df_subset.y_val[index]+0.3, s=value)