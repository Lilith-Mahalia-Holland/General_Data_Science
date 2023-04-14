import pandas as pd
import gensim.corpora as corpora
from pprint import pprint
import pickle
import re
import pyLDAvis.gensim_models
from gensim.models import coherencemodel

journal_df = pd.read_csv('./NIH/data/Cleaned_NIH.csv')

nested_rows = [['path', 'paper', 'author', 'text']]

# march over the entire dataframe so that text can be author information and text can be build per paper
for path in journal_df.path.unique():
    for paper in journal_df.loc[journal_df.path == path, 'paper_number'].unique():

        restructured_row = []
        index = -1

        restructured_row.append(path)
        restructured_row.append(paper)

        # extract author information
        if journal_df.loc[(journal_df.path == path) & (journal_df.paper_number == paper) & (journal_df.search_type == 'author'), 'text'].any():
            restructured_row.append(journal_df.loc[(journal_df.path == path) & (journal_df.paper_number == paper) & (journal_df.search_type == 'author'), 'text'].values[0])
        else:
            restructured_row.append('none')

        # extract text information, can throw error as some papers have no text
        try:
            restructured_row.append(journal_df.loc[(journal_df.search_type == 'none'), :].groupby(['path', 'paper_number'])['text'].apply('.'.join)[path][paper])
        except:
            restructured_row.append('none')

        nested_rows.append(restructured_row)

restructured_df = pd.DataFrame(nested_rows[1:], columns=nested_rows[0])

# remove rows missing text and correct paper numbering
restructured_df.drop(restructured_df.loc[restructured_df.text == 'none', :].index, inplace=True)
restructured_df.reset_index(drop=True, inplace=True)
restructured_df.paper = restructured_df.index

restructured_df['processed_text'] = restructured_df['text'].map(lambda x: re.sub('[.]', '\n', x))

import os
os.environ.update({'MALLET_HOME':r'E:/mallet-2.0.8/mallet-2.0.8/'})
mallet_path = r'E:/mallet-2.0.8/mallet-2.0.8/bin/mallet'

# get version 3.8.3 for mallet
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['om', 'buds', 'ombudsman', 'ombudsmen'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]
data = restructured_df.processed_text.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

num_topics = [10, 15, 20]
results_folder = './NIH/results/ldavis_prepared_'
# check if the file exists and create it if not
#
#
#

# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

import joblib

for num in num_topics:
    path = os.path.join(results_folder + str(num) + '.html')

    base_path = os.path.join(results_folder + str(num))
    with open(path, 'w'):
        pass
    ldamallet = gensim.models.wrappers.LdaMallet(
       mallet_path, corpus=corpus, num_topics=num, id2word=id2word
    )
    pprint(ldamallet.show_topics(formatted=False))

    # Visualize the topics
    #pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join(results_folder + str(num))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        # pyLDAvis 3.3.0
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet), corpus, id2word)
        with open(base_path, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(base_path, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, path)
    LDAvis_prepared
