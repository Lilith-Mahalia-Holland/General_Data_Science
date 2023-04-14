
# Author: Lilith Holland
# Make sure that box drive is installed and you have editor approval on the file.
# Make sure that the variables below are all configured to your current machine.
# This script will crash sometimes, I'm working on fixing this but it's a difficult problem.






import pandas as pd
import numpy as np
import os
import sys
import re
import time
import timeit





#file = 'C:/Users/jdhjr/Box/COVID-19 Raw Twitter JSONs/04-2020/2020-04-15-22.json'


# ALL OF THIS NEEDS TO BE SETUP FOR YOUR CURRENT SYSTEM
# Base path to the save and load locations, above any folders that are used
# Since this script modifies and creates files make sure you change the path's, otherwise the script will not work
load_base_path = "C:/Users/jdhjr/Box/COVID-19 Raw Twitter JSONs"
save_base_path = "C:/Users/jdhjr/Box/COVID-19 Flattened Twitter CSVs"
user_base_dir = "C:/Users/jdhjr"
# The last folder will not be done due to how python works, if you want to do 12 folders put 0 and 12 in.
folder_start = 2
folder_end = 3
all_folder = True


if not os.path.isdir(user_base_dir):
    sys.exit()


# I search for all of the json data by walking the load_path location
load_folder_name = next(os.walk(load_base_path), (None, None, []))[1]
save_folder_name = load_folder_name


# Combine the full load path with each individual folder name and append these to a new list
load_folder_path = []
save_folder_path = []
for i in range(0, len(load_folder_name)):
    load_folder_path.append(load_base_path + "/" + load_folder_name[i])
    save_folder_path.append(save_base_path + "/" + save_folder_name[i])


for i in range(0, len(save_folder_path)):
    if os.path.exists(save_base_path) and not os.path.exists(save_folder_path[i]):
        os.makedirs(save_folder_path[i])


# Walk the path of each folder and create a sub list of all file names for each folder name
# This loop is only keeping the names of each folder for saving reasons
save_file_path = []
load_file_path = []
save_file_name = []
for i in range(0, len(load_folder_path)):
    # Walk the contents of folder i in the load_path_temp list
    load_file_name = next(os.walk(load_folder_path[i]), (None, None, []))[2]
    inner_save_file_path = []
    inner_load_file_path = []
    inner_save_file_name = []
    # Go through all files and only keep files that are of type .json and append these files to the inner_path list
    for j in range(0, len(load_file_name)):
        if load_file_name[j].endswith(".json"):
            # Append paths together and create a list for names, this is all done so I can dynamically name and load
            # files
            inner_save_file_path.append(save_folder_path[i] + "/" + os.path.splitext(load_file_name[j])[0])
            inner_load_file_path.append(load_folder_path[i] + "/" + load_file_name[j])
            inner_save_file_name.append(os.path.splitext(load_file_name[j])[0])
    save_file_path.append(inner_save_file_path)
    load_file_path.append(inner_load_file_path)
    save_file_name.append(inner_save_file_name)


del(i, j, inner_load_file_path, inner_save_file_name, inner_save_file_path, load_file_name, load_folder_name,
    load_folder_path, save_folder_name)

listed_columns = [
    'created_at', 'id_str', 'text', 'truncated', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'user',
    'extended_tweet', 'entities', 'retweeted_status', 'quoted_status', 'screen_name', 'description', 'verified',
    'followers_count', 'friends_count', 'favourites_count', 'statuses_count', 'created_at', 'full_text', 'hashtags',
    'user_mentions', 'entities_hashtags', 'entities_user_mentions'
]
user_listed_columns = ['user_' + sub for sub in listed_columns]
extended_tweet_listed_columns = ['extended_tweet_' + sub for sub in listed_columns]
entities_listed_columns = ['entities_' + sub for sub in listed_columns]

def flatten_dict(df):
    df = df.explode().dropna(how='all')
    df = pd.DataFrame(df.values.tolist(), index=[df.index])
    df.reset_index(inplace=True)
    df = df.groupby('level_0').agg(list)
    return df

# Simple check to load all folders if required
if all_folder:
    folder_end = len(save_folder_path)
    folder_start = 0

for i in range(folder_start, folder_end):
    print("Starting folder {:d}\n".format(i + 1))
    # Run through all possible files in each folder
    for j in range(0, len(load_file_path[i])):
        # If a file exists make sure to not re calculate the file
        if not os.path.isfile(save_file_path[i][j] + ".zip"):
            # If the file does not exist make an empty data frame
            df = pd.DataFrame()
            print("Starting folder {:d}, file {:d}\n".format((i + 1), (j + 1)))
            # Read through the JSON in chunks of 10,000
            # There is some kind of bug with converting the slices into a list
            # for df in reader: lines = list(islice(self.data, self.chunksize))
            try:
                start_time = time.time()
                df = pd.read_json(load_file_path[i][j], lines=True, orient="columns", encoding="utf-8-sig")
                df.drop(columns=df.columns.difference(listed_columns), inplace=True)
                df.truncated = df.truncated.astype(bool)

                #####################################

                # speed this up by potentially using the flatten dict function
                # flatten the users column
                user_df = pd.json_normalize(df.user, sep="_")
                user_df.drop(columns=user_df.columns.difference(listed_columns), inplace=True)
                user_df.columns = user_df.add_prefix('user_').columns

                #####################################

                # same for this
                # flatten the extended_tweet column
                ex_tweet_df = pd.json_normalize(df.extended_tweet, sep="_")
                ex_tweet_df.drop(columns=ex_tweet_df.columns.difference(listed_columns), inplace=True)

                # extract extended tweet hashtags
                ex_tweet_hash_df = flatten_dict(ex_tweet_df.entities_hashtags)
                ex_tweet_hash_df.drop(columns=ex_tweet_hash_df.columns.difference(listed_columns), inplace=True)
                ex_tweet_hash_df.columns = ['entities_hashtags']

                # extract extended tweet mentions
                ex_tweet_ment_df = flatten_dict(ex_tweet_df.entities_user_mentions)
                ex_tweet_ment_df.drop(columns=ex_tweet_ment_df.columns.difference(listed_columns), inplace=True)
                ex_tweet_ment_df.columns = ex_tweet_ment_df.add_prefix('entities_user_mentions_').columns

                # remove hashtags and mentions columns from extended tweet
                ex_tweet_df.drop(columns=ex_tweet_df.columns.difference(['full_text']), inplace=True)

                # concatenate columns and remove redundant variables
                ex_tweet_df = pd.concat([ex_tweet_df, ex_tweet_hash_df, ex_tweet_ment_df], axis=1)
                ex_tweet_df.columns = ex_tweet_df.add_prefix('extended_tweet_').columns
                ex_tweet_df = ex_tweet_df.dropna(how='all')

                #####################################

                ent_df = pd.json_normalize(df.entities, sep="_")
                ent_df.drop(columns=ent_df.columns.difference(listed_columns), inplace=True)

                ent_hash_df = flatten_dict(ent_df.hashtags)
                ent_hash_df.drop(columns=ent_hash_df.columns.difference(listed_columns), inplace=True)
                ent_hash_df.columns = ['hashtags']

                ent_ment_df = flatten_dict(ent_df.user_mentions)
                ent_ment_df.drop(columns=ent_ment_df.columns.difference(listed_columns), inplace=True)
                ent_ment_df.columns = ent_ment_df.add_prefix('user_mentions_').columns

                ent_df = pd.concat([ent_hash_df, ent_ment_df], axis=1)
                ent_df.columns = ent_df.add_prefix('entities_').columns

                #####################################

                rtweet_df = pd.json_normalize(df.retweeted_status, sep="_")
                rtweet_df.drop(columns=rtweet_df.columns.difference(
                    listed_columns + user_listed_columns + extended_tweet_listed_columns
                    + entities_listed_columns), inplace=True)

                rtweet_ex_tweet_hash_df = flatten_dict(rtweet_df.extended_tweet_entities_hashtags)
                rtweet_ex_tweet_hash_df.drop(columns=rtweet_ex_tweet_hash_df.columns.difference(listed_columns),
                                             inplace=True)
                rtweet_ex_tweet_hash_df.columns = ['extended_tweet_entities_hashtags']

                rtweet_ex_tweet_ment_df = flatten_dict(rtweet_df.extended_tweet_entities_user_mentions)
                rtweet_ex_tweet_ment_df.drop(columns=rtweet_ex_tweet_ment_df.columns.difference(listed_columns),
                                             inplace=True)
                rtweet_ex_tweet_ment_df.columns = rtweet_ex_tweet_ment_df.add_prefix(
                    'extended_tweet_entities_user_mentions_').columns

                rtweet_ent_hash_df = flatten_dict(rtweet_df.entities_hashtags)
                rtweet_ent_hash_df.drop(columns=rtweet_ent_hash_df.columns.difference(listed_columns), inplace=True)
                rtweet_ent_hash_df.columns = ['entities_hashtags']

                rtweet_ent_ment_df = flatten_dict(rtweet_df.entities_user_mentions)
                rtweet_ent_ment_df.drop(columns=rtweet_ent_ment_df.columns.difference(listed_columns), inplace=True)
                rtweet_ent_ment_df.columns = rtweet_ent_ment_df.add_prefix('entities_user_mentions_').columns

                rtweet_df.drop(columns=['extended_tweet_entities_hashtags', 'extended_tweet_entities_user_mentions',
                                        'entities_hashtags', 'entities_user_mentions'], inplace=True)
                rtweet_df = pd.concat([rtweet_df, rtweet_ex_tweet_hash_df, rtweet_ex_tweet_ment_df, rtweet_ent_hash_df,
                                       rtweet_ent_ment_df], axis=1)
                rtweet_df.columns = rtweet_df.add_prefix('retweeted_status_').columns
                rtweet_df = rtweet_df.dropna(how='all')

                #####################################

                qtweet_df = pd.json_normalize(df.quoted_status, sep="_")
                qtweet_df.drop(columns=qtweet_df.columns.difference(
                    ['text', 'truncated'] + user_listed_columns + extended_tweet_listed_columns
                    + entities_listed_columns), inplace=True)

                qtweet_ex_tweet_hash_df = flatten_dict(qtweet_df.extended_tweet_entities_hashtags)
                qtweet_ex_tweet_hash_df.drop(columns=qtweet_ex_tweet_hash_df.columns.difference(listed_columns),
                                             inplace=True)
                qtweet_ex_tweet_hash_df.columns = ['extended_tweet_entities_hashtags']

                qtweet_ex_tweet_ment_df = flatten_dict(qtweet_df.extended_tweet_entities_user_mentions)
                qtweet_ex_tweet_ment_df.drop(columns=qtweet_ex_tweet_ment_df.columns.difference(listed_columns),
                                             inplace=True)
                qtweet_ex_tweet_ment_df.columns = qtweet_ex_tweet_ment_df.add_prefix(
                    'extended_tweet_entities_user_mentions_').columns

                qtweet_ent_hash_df = flatten_dict(qtweet_df.entities_hashtags)
                qtweet_ent_hash_df.drop(columns=qtweet_ent_hash_df.columns.difference(listed_columns), inplace=True)
                qtweet_ent_hash_df.columns = ['entities_hashtags']

                qtweet_ent_ment_df = flatten_dict(qtweet_df.entities_user_mentions)
                qtweet_ent_ment_df.drop(columns=qtweet_ent_ment_df.columns.difference(listed_columns), inplace=True)
                qtweet_ent_ment_df.columns = qtweet_ent_ment_df.add_prefix('entities_user_mentions_').columns

                qtweet_df.drop(columns=['extended_tweet_entities_hashtags', 'extended_tweet_entities_user_mentions',
                                        'entities_hashtags', 'entities_user_mentions'], inplace=True)
                qtweet_df = pd.concat([qtweet_df, qtweet_ex_tweet_hash_df, qtweet_ex_tweet_ment_df, qtweet_ent_hash_df,
                                       qtweet_ent_ment_df], axis=1)
                qtweet_df.columns = qtweet_df.add_prefix('quoted_status_').columns
                qtweet_df = qtweet_df.dropna(how='all')

                #####################################

                df.drop(columns=['user', 'extended_tweet', 'entities', 'retweeted_status', 'quoted_status'],
                        inplace=True)
                df = pd.concat([df, user_df, ex_tweet_df, ent_df, rtweet_df, qtweet_df], axis=1)
            except:
                print("Error in folder {:d}, file {:d}\n".format((i + 1), (j + 1)))
            else:
                # create a zip file for each data frame, I would prefer to make them share but that has issues atm
                compression_opts = dict(method="zip", archive_name=save_file_name[i][j] + ".csv")
                # Save the csv into the zip file
                df.to_csv(save_file_path[i][j] + ".zip", index=False, compression=compression_opts)
                print("Finished folder {:d}, file {:d}\n".format((i + 1), (j + 1)))
                print("--- %s seconds ---\n" % (time.time() - start_time))
        else:
            print("Folder {:d}, file {:d} has already been done\n".format((i + 1), (j + 1)))
    print("Finished folder {:d}\n".format(i + 1))


# Expand the read out to be easier to glance at




