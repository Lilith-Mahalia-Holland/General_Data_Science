import numpy as np
import polars as pl
import pandas as pd
import itertools
import scipy

# type, name, ap, accuracy
gear = {'earring': {'dawn': [14, 38],
                    'disto': [21, 16],
                    'debo': [19, 12],
                    'tungrad': [17, 12]},
        'ring': {'crescent': [20, 12],
                 'tungrad': [21, 12],
                 'ominous': [18, 28]},
        'necklace': {'debo': [40, 24],
                     'lunar': [31, 52],
                     'ogre': [35, 24]},
        'belt': {'turo': [17, 34],
                 'debo': [24, 12],
                 'tungrad': [21, 12],
                 'vultara': [20, 12]},
        'off-hand': {'nouver': [47, 12],
                     'saiyer': [17, 108]}}

# earring, ring, necklace, belt, off-hand
accessory_slots = [2, 2, 1, 1, 1]

bracket = [[0, 0], [100, 5], [140, 10], [170, 15], [184, 20], [209, 30], [235, 40], [245, 48], [249, 57], [253, 69],
           [257, 83], [261, 101], [265, 122], [269, 137], [273, 142], [277, 148], [281, 154], [285, 160], [289, 167],
           [293, 174], [297, 181], [301, 188], [305, 196], [309, 200], [316, 203], [323, 205], [330, 207], [340, 210]]

def ap_bracket_check(ap, bracket, single=False):
    all_bonus = []
    for item in ap:
        for value in bracket:
            if value[0] <= item:
                bonus = value[1]
        all_bonus.append(bonus)
    if single:
        return all_bonus[0]
    else:
        return all_bonus

def reverse_ap_bracket(ap, bracket, single=False):
    all_base = []
    for item in ap:
        for value in bracket:
            if value[1] + value[0] <= item:
                base = value[0]
        all_base.append(base)
    if single:
        return all_base[0]
    else:
        return all_base

def flat2gen(alist):
  for item in alist:
    if isinstance(item, list):
      for subitem in item:
          yield subitem
    else:
      yield item





#base_ap = 189
base_ap = 149

buff_acc = 566
base_ap = 13
base_dp = 9

# make a function to convert this back and forth
# I don't want to keep converting stuff by hand
target_ap = 316
bracket_ap = target_ap + ap_bracket_check([target_ap], bracket, True)






gear_name = []
for index, name in enumerate(gear):
    for item in range(accessory_slots[index]):
        gear_name.append(name)



accessory_list = []
for index, accessory in enumerate(gear):
    temp = []
    if accessory_slots[index] <= 1:
        for item in gear[accessory]:
            temp.append(item)
    else:
        for item in itertools.combinations_with_replacement(gear[accessory], accessory_slots[index]):
            temp.append(item)
    accessory_list.append(temp)

combinations = [p for p in itertools.product(*accessory_list)]

combinations_accuracy = []
combinations_ap = []
flattened_combinations = []

for group in combinations:
    temp = []
    for item in group:
        if isinstance(item, tuple):
            for subitem in item:
                temp.append(subitem)
        else:
            temp.append(item)
    group = temp
    ap = 0
    accuracy = 0
    for index, item in enumerate(group):
        ap = ap + gear[gear_name[index]][item][0]
        accuracy = accuracy + gear[gear_name[index]][item][1]
    flattened_combinations.append(group)
    combinations_ap.append(ap)
    combinations_accuracy.append(accuracy)

ap = np.array(combinations_ap) + base_ap

total_ap = np.array(ap_bracket_check(ap, bracket)) + ap



combinations_df = pd.DataFrame({'combinations': flattened_combinations,
                                'total_ap': total_ap,
                                'accuracy': combinations_accuracy})


old_minimum = np.Inf
target_index = None

# This selects the first item that meets the conditions
# if you want to evaluate the true balance value of a buff, maybe normalize it's  balance value to it's total buff value divided by the price
for index, item in enumerate(total_ap):
    diff = abs(item - combinations_accuracy[index])
    if diff < old_minimum and item > bracket_ap:
        old_minimum = diff
        target_index = index

combinations_df.total_ap = reverse_ap_bracket(combinations_df.total_ap, bracket)

print(combinations_df.iloc[[target_index]])
print(combinations_df.combinations[target_index])

combinations_df = combinations_df.sort_values(by=['total_ap', 'accuracy'])



#temp_df = combinations_df.sort_values(by=['total_ap', 'accuracy'])
#acc_test = max(temp_df[temp_df.total_ap >= target_ap + ap_bracket_check([target_ap], bracket, True)].accuracy)
#temp_df[(temp_df.total_ap >= target_ap + ap_bracket_check([target_ap], bracket, True)) & (temp_df.accuracy >= acc_test)]
