import numpy as np
import pandas as pd
import itertools
import time

start_time = time.time()
nom = [1, 5, 6, 2, 3, 5, 9, 1, 2, 2, 5, 1, 3]
denom = [1, 6, 5, 3, 2, 9, 5, 2, 1, 5, 2, 3, 1]
ratio_string = ['1/1', '5/6', '6/5', '2/3', '3/2', '5/9', '9/5', '1/2', '2/1', '2/5', '5/2', '1/3', '3/1']
test = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "a", 'b', 'c']
gears = 9
under_gears = 0
m_error = 0.05

target = []
for i in range(-under_gears, (gears+1-under_gears)):
    target.append((84*i/gears+36)/36)

start_time = time.time()
final_index = []
final_ratio = []
# the range should not include gear 0 as it has no value in this comparison 
for i in range(2, gears+1):
    temp = []
    ratio = []
    for j in itertools.combinations_with_replacement(range(13), i):
        k_nom = 1
        k_denom = 1
        for k in j:
            k_nom = nom[k] * k_nom
            k_denom = denom[k] * k_denom
        ratio.append(k_nom/k_denom)
        temp.append(j)
    final_index.append(temp)
    final_ratio.append(ratio)
del(ratio, temp)
print("Total time: " + str(time.time()-start_time))

start_time = time.time()
index_df = pd.DataFrame(final_index)
index_df = index_df.transpose()
index_df = index_df.melt()
index_df = index_df.rename(columns={"variable": "index_variable",
                                    "value": "index_value"})

ratio_df = pd.DataFrame(final_ratio)
ratio_df = ratio_df.transpose()
ratio_df = ratio_df.melt()
ratio_df = ratio_df.rename(columns={"variable": "ratio_variable",
                                    "value": "ratio_value"})
print("Total time: " + str(time.time()-start_time))

cleaned_df = pd.concat([index_df, ratio_df], axis=1)
del(ratio_df, index_df, final_index, final_ratio)

start_time = time.time()
cleaned_df = cleaned_df[cleaned_df.ratio_value.between(target[0]-m_error, target[-1]+m_error)]
print("Total time: " + str(time.time()-start_time))


#final_df = final_df.drop_duplicates('ratio_value')
cleaned_df = cleaned_df.drop(columns='ratio_variable')
cleaned_df = cleaned_df.sort_values('ratio_value')



def filter_fn(row, target):
    if (target - 0.02) < row['ratio_value'] < (target + 0.02):
        return True
    else:
        return False

final_df = pd.DataFrame(columns=['index_variable', 'index_value', 'ratio_value'])
for index, i in enumerate(target):
    temp_df = cleaned_df[cleaned_df.apply(filter_fn, target=i, axis=1)]
    temp_df.insert(3, "group_index", index, True)
    final_df = final_df.append(temp_df, ignore_index=True)

#del(temp_df)
#del(cleaned_df)


final_df.index_value = final_df.index_value.apply(lambda x: x+(gears-len(x))*(0,) if len(x) < gears else x)
#final_df['ratio_index'] = list(final_df.ratio_value + final_df.index_value)



print("Total time: " + str(time.time()-start_time))


























# nothing, normal, boss, mission, pvp
score = [0, 40, 100, 100, 100]
actual_max = max(score)*14
desired_max = max(score)*13
desired_min = score[1]*3
iteration = []
for item in itertools.combinations_with_replacement(range(5), 14):
    iteration.append(item)

cleaned_iteration = []
for item in iteration:
    try:
        if{i:item.count(i) for i in item}[2] <= 2:
            cleaned_iteration.append(item)
    except:
        pass

final_list = []
for item in cleaned_iteration:
    inner_list = []
    for element in item:
        inner_list.append(score[element])
    final_list.append(sum(inner_list))

final_list.sort()

import matplotlib.pyplot as plt

#lower_score = np.linspace(score[1]*3, score[1]*13, 3)
#upper_score = np.linspace(score[1]*13, score[3]*12+score[2]*1, 8)
#lower_score = 400 - np.geomspace(score[1]*3, score[1]*13, 3)
#upper_score = 1325 - np.geomspace(score[1]*13, score[3]*12+score[2]*1, 8)
#score_range = lower_score[1:].tolist() + upper_score.tolist()

score_range = desired_max + desired_min - np.geomspace(score[1]*3, score[3]*12+score[2]*1, 10)
score_range = [int(x) for x in score_range]
score_range = [(x/max(final_list))*len(cleaned_iteration) for x in score_range]

# make a non-linear function that gets more dense the closer to max
plt.plot(final_list)
for item in score_range:
    plt.axvline(x=item)
plt.xticks(np.arange(0, (max(final_list)+1)/max(final_list)*len(cleaned_iteration), 50))
plt.yticks(np.arange(0, max(final_list)+1, 50))
plt.show()