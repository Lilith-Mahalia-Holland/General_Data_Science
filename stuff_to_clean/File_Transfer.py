import os
import shutil


source_file_location = "C:/Users/jdhjr/Box/COVID-19 Flattened Twitter CSVs"
target_file_location = "F:/Machine Learning/twitter_csv's"


source_path = next(os.walk(source_file_location), (None, None, []))[1]
temp_source_path = []
for i in range(0, len(source_path)):
    temp_source_path.append(source_file_location + "/" + source_path[i])


source_path = []
source_name = []
for i in range(0, len(temp_source_path)):
    file_name = next(os.walk(temp_source_path[i]), (None, None, []))[2]
    temp_inner_source_path = []
    temp_inner_source_name = []
    for j in range(0, len(file_name)):
        if file_name[j].endswith(".zip"):
            temp_inner_source_path.append(temp_source_path[i] + "/" + os.path.splitext(file_name[j])[0] + ".zip")
            temp_inner_source_name.append(os.path.splitext(file_name[j])[0] + ".csv")
    source_path.append(temp_inner_source_path)
    source_name.append(temp_inner_source_name)


for i in range(0, len(source_path)):
    for j in range(0, len(source_path[i])):
        if not os.path.exists(target_file_location + "/" + source_name[i][j]):
            try:
                print("Transferring folder:%d file:%d" % (i, j))
                shutil.unpack_archive(source_path[i][j], target_file_location, "zip")
            except:
                print("Error in transferring")
        else:
            print("folder:%d file:%d already exist")
