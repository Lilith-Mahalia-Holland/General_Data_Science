import pandas as pd
import numpy as np
author_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1igglfcvDxJze9y_g---8JFPvkUutmYINsIRcT3TLOfE/export?gid=1618637841&format=csv')
submissions_df = pd.read_csv('../stuff_to_clean/submissions.xlsx - Submissions.csv')
section_df = pd.read_csv('../stuff_to_clean/Section titles and numbers - Sheet1.csv')


submissions_df = submissions_df.drop(index=[8, 9])
submissions_df = submissions_df.reset_index()
submissions_df = submissions_df.drop(columns='index')


section_df.Authors = section_df.Authors.str.replace("Chairs: ", "")


#temp_df = submissions_df.copy()
temp_df1 = pd.DataFrame()
#temp_df2 = pd.DataFrame()
for i in range(0, len(submissions_df)):
    temp = section_df.Authors.str.match(submissions_df.Authors[i], False)
    temp_df1 = temp_df1.append({"authors": submissions_df.Authors[i],
                                "title": submissions_df.Title[i],
                                "track": submissions_df.Track[i],
                                "section_name": np.nan if section_df["Section Title"][temp].any() == False else section_df["Section Title"][temp],
                                "section_number": np.nan if section_df["Section Number"][temp].any() == False else section_df["Section Number"][temp]},
                                ignore_index=True)

#    temp = section_df.Title.str.match(submissions_df.Title[i], False, na=False)
#    temp_df2 = temp_df2.append({"title": submissions_df.Title[i],
#                                "section_name": np.nan if section_df["Section Title"][temp].any() == False else section_df["Section Title"][temp],
#                                "section_number": np.nan if section_df["Section Number"][temp].any() == False else section_df["Section Number"][temp]},
#                                ignore_index=True)


working_df = temp_df1.dropna()
working_df["submission number"] = submissions_df['#']
working_df = working_df.explode(['section_name', 'section_number'])
working_df.authors = working_df.authors.str.split(pat=' and |, ')
working_df['author_sequence'] = working_df.authors.map(len)
working_df.author_sequence = working_df.author_sequence.map(np.arange)
working_df = working_df.explode(['authors', 'author_sequence'])
working_df.author_sequence = working_df.author_sequence + 1
working_df = working_df.reset_index()
working_df = working_df.drop(columns='index')


author_df['full_name'] = author_df['first name'] + " " + author_df['last name']
author_df = author_df[author_df['full_name'].isin(working_df.authors)]
working_df = working_df[working_df.authors.isin(author_df.full_name)]


final_df = author_df.merge(working_df, how='left', left_on='full_name', right_on='authors')
final_df.to_csv('csv_columns.csv')






import pandas as pd
data1 = pd.read_csv('https://docs.google.com/spreadsheets/d/1v8Dn5S-iesNbijkFcWeXulQ3nmHwWSAthqg3I3lSp_U/export?gid=13538904&format=csv')
data2 = pd.read_csv('https://docs.google.com/spreadsheets/d/1igglfcvDxJze9y_g---8JFPvkUutmYINsIRcT3TLOfE/export?gid=1618637841&format=csv')
used_data1 = data1[data1.role == 'PC member']
used_data2 = data2


used_data1.fillna('', inplace=True)
used_data2.fillna('', inplace=True)

#mixed_data = used_data2[used_data2["person #"].isin(used_data1["person #"])]

final_data1 = pd.DataFrame()
final_data1 = used_data1["first name"] + " " + used_data1["last name"]
final_data1 = final_data1 + ', ' + used_data1['affiliation']
final_data1 = final_data1 + ', ' + used_data1['country']

final_data2 = pd.DataFrame()
final_data2 = used_data2["first name"] + " " + used_data2["last name"]
final_data2 = final_data2 + ', ' + used_data2['affiliation']
final_data2 = final_data2 + ', ' + used_data2['country']

mixed_data = pd.concat([final_data1, final_data2])


#final_data = final_data.sort_values()
mixed_data = mixed_data.drop_duplicates()
mixed_data = mixed_data.reset_index()
mixed_data = mixed_data[0]


final_data1 = final_data1.reset_index()
final_data1 = final_data1[0]


with open("../stuff_to_clean/Program committee.txt", "w", encoding="utf-8") as text_file:
    for i in range(0, final_data1.shape[0]):
        text_file.write(final_data1[i] + "\n")





import pandas as pd
data = pd.read_csv('https://docs.google.com/spreadsheets/d/1v8Dn5S-iesNbijkFcWeXulQ3nmHwWSAthqg3I3lSp_U/export?gid=13538904&format=csv')
used_data = data[data.role == 'track chair']

used_data.fillna('', inplace=True)

final_data = pd.DataFrame()
final_data = used_data["first name"] + " " + used_data["last name"]
final_data = final_data + ', ' + used_data['affiliation']
final_data = final_data + ', ' + used_data['country']

#final_data = final_data.sort_values()
final_data = final_data.drop_duplicates()
final_data = final_data.reset_index()
final_data = final_data[0]

with open("../stuff_to_clean/Track Chairs.txt", "w") as text_file:
    for i in range(0, final_data.shape[0]):
        text_file.write(final_data[i] + "\n")