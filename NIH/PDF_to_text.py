# Author: Lilith Holland

import PyPDF2
import pandas as pd
import re
import os
import sys
import string
from collections import Counter


base = "F:/Machine Learning/PDF's"
if not os.path.exists(base):
    sys.exit()

file_path = []
for path in os.walk(base):
    if len(path[1]) == 0:
        root, _, files = path
        for file in files:
            file_path.append(os.path.join(root, file))

# ----------------------------------------------------------------------------------------------------------------------
# Written by D Greenberg https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
# -*- coding: utf-8 -*-
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

# ----------------------------------------------------------------------------------------------------------------------
# Written by Darius Bacon
# Edited by kumardeepakr3
# https://stackoverflow.com/questions/195010/how-can-i-split-multiple-joined-words
# Dictionary from https://github.com/dwyl/english-words

def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

def word_prob(word):
    return dictionary[word] / total

def words(text):
    return re.findall('[a-z]+', text.lower())

with open('./Dictionaries/words_alpha.txt') as dictionary:
    dictionary = Counter(words('\n'.join([w for w in dictionary.read().split() if len(w) > 1])))
    #dictionary = Counter(words('\n'.join([w for w in dictionary.read().split()])))
    dictionary['ombudsman', 'ombudsmen', 'ombud'] = 1
    max_word_length = max(map(len, dictionary))
    total = float(sum(dictionary.values()))

#  ----------------------------------------------------------------------------------------------------------------------

def all_equal_ivo(lst):
    return not lst or lst.count(lst[0]) == len(lst)

#  ----------------------------------------------------------------------------------------------------------------------

def join_words(text):
    string = []
    for value in text.split():
        if not dictionary[value] > 0:
            split_text, score = viterbi_segment(value)
            if score != 0.0:
                for sub_value in split_text:
                    string.append(sub_value)
        else:
            string.append(value)

    return ' '.join(string)

#  ----------------------------------------------------------------------------------------------------------------------

def count_words(text):
    for value in text.split():
        if dictionary[value] > 0:
            dictionary[value] += 1
    return text

#  ----------------------------------------------------------------------------------------------------------------------

def text_cleanup(text):
    text = str(text)
    text = re.sub(r'[0-9]', '', text)
    text = str.lower(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = str.strip(text)
    return text

#  ----------------------------------------------------------------------------------------------------------------------

def min_words(text):
    if len(text.split()) < 6:
        return ''
    else:
        return text

#  ----------------------------------------------------------------------------------------------------------------------

#text_df = pd.Series(dtype='O')
exclusion_regex = '^(?!.*ABSTRACT)(?!.*ABOUT THE AUTHOR)(?!.*FIGURE)(?!.*TABLE)(?!.*KEYWORDS)(?!.*KEY WORDS)'
# catch side papers on end of document
# the logic for authors needs to be expanded, look into title test that doesn't rely on an author
# set a flag to ignore all authors until first author abstract pair after which take all authors
regex_list = ['[A-Z]\s[A-Z]\s[A-Z]\s\s\s[A-Z]\s[A-Z]\s[A-Z]\s',                                                         # Author check
              f'{exclusion_regex}[A-Z]{{2,}}\W+(?:\w+\W+){{0,10}}?[A-Z]{{2,}}$|{exclusion_regex}[A-Z]{{6,}}$',          # Title check
              'FIGURE|TABLE',                                                                                           # Figure check
              '[Vv]olume\s[0-9],\s[Nn]umber\s[0-9]',                                                                    # Header
              'ABSTRACT',                                                                                               # Abstract
              'ABOUT THE AUTHOR',
              'KEYWORDS|KEY WORDS'
              ]

#  ----------------------------------------------------------------------------------------------------------------------

final_df = pd.DataFrame()

for path_index, path in enumerate(file_path):
    # set conditions
    print(path_index, path)
    # file 0, 1-33, and 34+ all need to be processed differently
    set_1 = path_index == 0
    set_2 = path_index != 0
    set_3 = path_index >= 34

    regex_result_list = []
    # extract all of the text and mark it with a label
    with open(path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj, strict=False)
        for page_index, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text().split('\n')
            for block_index, block in enumerate(page_text):
                # file 8 has an issue with formatting and needs to be converted
                if path_index == 8:
                    block = block.upper()
                regex_sub_list = []
                for regex_index, regex in enumerate(regex_list):
                    block_search = re.search(regex, block)
                    # This is written in python 3.9, there are no switch statements yet
                    if block_search is None:
                        block_search = 'None'
                    elif regex_index == 0:
                        block_search = 'Author'
                    elif regex_index == 1:
                        block_search = 'SubTitle'
                    elif regex_index == 2:
                        block_search = 'Figure/Table'
                    elif regex_index == 3:
                        block_search = 'Header'
                    elif regex_index == 4:
                        block_search = 'Abstract'
                    elif regex_index == 5:
                        block_search = 'About_author'
                    elif regex_index == 6:
                        block_search = 'Keywords'
                    else:
                        block_search = 'Error'
                    regex_sub_list.append([page_index, block_index, block_search, block])
                # remove duplicate entries since a text can be marked multiple times
                if all_equal_ivo([element[2] for element in regex_sub_list]):
                    regex_result_list.append(regex_sub_list[0])
                else:
                    for element in regex_sub_list:
                        if element[2] != 'None':
                            regex_result_list.append(element)

    text_df = pd.DataFrame(regex_result_list, columns=['page_index', 'block_index', 'search_type', 'text'])


    author_flag = False
    about_author_flag = False
    first_author_flag = False
    table_flag = False
    abstract_flag = False
    keyword_flag = False

    author_index = 0
    about_author_index = 0
    table_index = 0
    abstract_index = 0
    keyword_index = 0

    removed_columns = []

    if set_1:
        # mark columns for removal, do not remove on active set
        for index, r_type in enumerate(text_df.search_type):
            if r_type == 'Author':
                #if not first_author_flag: # remove all rows before first author must be done in second pass
                #in second pass also add the title marker and remove blank text rows
                # also remove from last about author till end
                #    first_author_flag = True
                #    first_author_range = range(index)
                #    if len(first_author_range) != 0:
                #        text_df.drop(text_df.index[first_author_range], axis=0, inplace=True)
                if author_flag: # if author flag is set already and new author is found mark old one
                    text_df.loc[author_index, 'search_type'] = 'None'
                    author_index = index
                else: # If first author is found set flag
                    author_flag = True
                    author_index = index
                if about_author_flag: # if author is found and about author flagged remove in between and flip flag
                    about_author_range = range(about_author_index, index-1) # Title is before author but cannot be marked yet
                    # combine variables
                    #if len(about_author_range) != 0:
                    removed_columns.append(text_df.index[about_author_range])
                    #    text_df.drop(text_df.index[about_author_index], axis=0, inplace=True)
                    about_author_flag = False
            elif r_type == 'Abstract' and author_flag: # if abstract found and author flag set change flag and remove between
                author_flag = False
                author_range = range(author_index+1,index)
                if len(author_range) != 0:
                    removed_columns.append(text_df.index[author_range])
                    #text_df.drop(text_df.index[author_range], axis=0, inplace=True)
            elif r_type != 'None' and author_flag: # if author flag is set and some marker found mark flag setter
                author_flag = False
                text_df.loc[author_index, 'search_type'] = 'None'

            elif r_type == 'About_author'and not about_author_flag: # if about author found and flag not set set flag
                about_author_flag = True
                about_author_index = index

            elif (r_type == 'SubTitle' or r_type == 'Figure/Table') and table_flag:
                # combine variables
                table_range = range(table_index, index)
                #if len(table_range) != 0:
                # add check for figure followed by figure
                removed_columns.append(text_df.index[table_range])
                    #text_df.drop(text_df.index[table_range], axis=0, inplace=True)
                if r_type != 'Figure/Table':
                    table_flag = False

            elif r_type == 'Figure/Table':
                table_flag = True
                table_index = index
    elif set_2:
        for index, r_type in enumerate(text_df.search_type):
            if r_type == 'Abstract':
                #if not first_author_flag: # remove all rows before first author must be done in second pass
                #in second pass also add the title marker and remove blank text rows
                # also remove from last about author till end
                #    first_author_flag = True
                #    first_author_range = range(index)
                #    if len(first_author_range) != 0:
                #        text_df.drop(text_df.index[first_author_range], axis=0, inplace=True)
                if abstract_flag: # if author flag is set already and new author is found mark old one
                    text_df.loc[abstract_index, 'search_type'] = 'None'
                    abstract_index = index
                else: # If first author is found set flag
                    abstract_flag = True
                    abstract_index = index
                if about_author_flag: # if author is found and about author flagged remove in between and flip flag
                    about_author_range = range(about_author_index, index) # Title is before author but cannot be marked yet
                    # combine variables
                    #if len(about_author_range) != 0:
                    removed_columns.append(text_df.index[about_author_range])
                    #    text_df.drop(text_df.index[about_author_index], axis=0, inplace=True)
                    about_author_flag = False
            elif r_type == 'Keywords' and abstract_flag: # if abstract found and author flag set change flag and remove between
                abstract_flag = False
                abstract_range = range(abstract_index+1,index)
                if len(abstract_range) != 0:
                    removed_columns.append(text_df.index[abstract_range])
                    #text_df.drop(text_df.index[author_range], axis=0, inplace=True)
            elif r_type != 'None' and abstract_flag: # if author flag is set and some marker found mark flag setter
                abstract_flag = False
                text_df.loc[abstract_index, 'search_type'] = 'None'

            elif r_type == 'About_author'and not about_author_flag: # if about author found and flag not set set flag
                about_author_flag = True
                about_author_index = index

            elif (r_type == 'SubTitle' or r_type == 'Figure/Table') and table_flag:
                # combine variables
                table_range = range(table_index, index)
                #if len(table_range) != 0:
                # add check for figure followed by figure
                removed_columns.append(text_df.index[table_range])
                    #text_df.drop(text_df.index[table_range], axis=0, inplace=True)
                if r_type != 'Figure/Table':
                    table_flag = False

            elif r_type == 'Figure/Table':
                table_flag = True
                table_index = index

    # Correct the author and abstract text
    text_df.loc[text_df.search_type == 'Author', 'text'] = text_df.text[text_df.search_type == 'Author'].apply(lambda x: re.sub('([A-Z])ABSTRACT$', '\g<1>', x))
    text_df.loc[text_df.search_type == 'Abstract', 'text'] = text_df.text[text_df.search_type == 'Abstract'].apply(lambda x: re.sub('^(.*?)[A-Z]ABSTRACT$', 'ABSTRACT', x))

    # Remove all marked elements and correct index
    text_df.drop([element for inner_list in removed_columns for element in inner_list], axis=0, inplace=True)
    text_df.reset_index(drop=True, inplace=True)

    # Remove all pages before first paper and label papers
    if set_1:
        text_df.drop(text_df.index[range(text_df.search_type[text_df.search_type == 'Author'].index[0]-1)], axis=0, inplace=True)
    elif set_3:
        text_df.loc[text_df.loc[text_df.search_type == 'Keywords', 'text'].index + 1, 'text'] = text_df.loc[text_df.loc[text_df.search_type == 'Keywords', 'text'].index + 1, 'text'] + '.'
    elif set_2:
        text_df.drop(text_df.index[range(text_df.search_type[text_df.search_type == 'Abstract'].index[0] - 1)], axis=0, inplace=True)
        text_df.loc[text_df.loc[text_df.search_type == 'Keywords', 'text'].index+1, 'text'] = text_df.loc[text_df.loc[text_df.search_type == 'Keywords', 'text'].index+1, 'text'] + '.'

    # Remove all headers and correct index
    text_df.drop(text_df[text_df.search_type == 'Header'].index, axis=0, inplace=True)
    text_df.reset_index(drop=True, inplace=True)

    # Label title before each author
    # This has no use as it only works on the first paper
    # text_df.loc[text_df.loc[text_df.search_type == 'Author'].index-1, 'search_type'] = 'Title'


    # group sentences into blocks
    count = -1
    group_index = []
    type_flag = False

    for type in text_df.search_type:
        if type != 'None':
            count += 1
            group_index.append(count)
            type_flag = False
        elif not type_flag:
            count += 1
            group_index.append(count)
            type_flag = True
        else:
            group_index.append(count)
    text_df['group_index'] = group_index

    # Compress the text into subject blocks
    text_df = text_df.loc[:, ['search_type', 'text', 'group_index']].groupby('group_index').agg(''.join)
    text_df.loc[text_df.search_type.str.contains('None'), 'search_type'] = 'None'

    # Mark each paper in each document
    paper_marker = [None] * len(text_df)
    previous_value = 0
    if set_1:
        for index, value in enumerate(text_df.loc[text_df.search_type == 'Author', 'text'].index.tolist()+[len(text_df)]):
            paper_marker[previous_value:value] = [index] * len(paper_marker[previous_value:value])
            previous_value = value
    elif set_2:
        for index, value in enumerate(text_df.loc[text_df.search_type == 'Abstract', 'text'].index.tolist()+[len(text_df)]):
            paper_marker[previous_value:value] = [index] * len(paper_marker[previous_value:value])
            previous_value = value
    text_df['paper_number'] = paper_marker

    # separate text into sentences
    text_df.loc[text_df.search_type == 'None', 'text'] = text_df.loc[text_df.search_type == 'None', 'text'].map(split_into_sentences)

    # Remove all index past last author
    try:
        text_df.drop(range(text_df.loc[text_df.search_type == 'About_author', 'text'].index[-1], text_df.index[-1] + 1), inplace=True)
    except:
        print('Cannot find end')
    text_df = text_df.explode('text')

    # Clean the text first pass
    text_df.loc[:, ['search_type', 'text']] = text_df.loc[:, ['search_type', 'text']].applymap(text_cleanup)
    text_df.reset_index(inplace=True)

    # Correct joined words
    text_df.loc[text_df.search_type == 'none', 'text'] = text_df.loc[text_df.search_type == 'none', 'text'].map(count_words)
    text_df.loc[text_df.search_type == 'none', 'text'] = text_df.loc[text_df.search_type == 'none', 'text'].map(join_words)

    # Clean the text second pass
    # Remove single letters
    text_df.loc[text_df.search_type == 'none', 'text'] = text_df.loc[text_df.search_type == 'none', 'text'].map(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    # Remove sentences with fewer than 6 words
    text_df.loc[text_df.search_type == 'none', 'text'] = text_df.loc[text_df.search_type == 'none', 'text'].map(min_words)
    # Cleanup again
    text_df.loc[text_df.search_type == 'none', 'text'] = text_df.loc[text_df.search_type == 'none', 'text'].map(text_cleanup)
    # Drop blank rows
    text_df.drop(text_df[text_df.loc[:, 'text'] == ''].index.tolist(), inplace=True)
    # Correct the index
    text_df.reset_index(drop=True, inplace=True)

    # mark each path
    text_df['path'] = path_index
    final_df = pd.concat([final_df, text_df], ignore_index=True)

#final_df.to_csv('Cleaned_NIH.csv', index=False)