import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os

# old = pd.read_csv("/Users/luisabrigo/HumanProgress/folder/old.csv", header=0)
# new = pd.read_csv("/Users/luisabrigo/HumanProgress/folder/new.csv", header=0)
#
# old = old.transpose()
# old.reset_index(level=0, inplace=True)
#
# old = old.rename(columns=old.iloc[0]).drop(old.index[0])
# old.reset_index(inplace=True)
# old = old.drop(['index'], axis=1)



# old = old.drop([1, 0])
# old = old.T
# old = old.set_index([np.arange(len(old)), old.index])



# print(old.index)
# print(new.index)
#
# print(len(old))
#

# nums = []
# for i in range(162):
#     i = str(i)
#     i = i.zfill(3)
#     nums.append(i)
#
# controles = dict(zip(nums, countries))

import json
# A dictionary of student names and their score
# Print contents of dict in json like format
# print(json.dumps(controles, indent=4))

directory = "/Users/luisabrigo/HumanProgress/all_indicators"
indicators = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # print(filename)
        short_name = filename[:-4]
        short_name = short_name.replace("-", " ")
        name = short_name.title()
        # print(name)
        indicators.append(str(name))


# print(indicators)

nums = []
for i in range(len(indicators)):
    i = str(i)
    i = i.zfill(4)
    nums.append(i)

controles = dict(zip(nums, indicators))
# print(controles)
json_string = json.dumps(controles, indent=4, ensure_ascii=False).encode('utf8')
# print(json_string.decode())

data = pd.read_csv("/Users/luisabrigo/HumanProgress/result_all_indicators.csv", header=0)

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


for i in data.columns[2:]:
    k = i[:-5]
    for j in indicators:
        res = similar(j, k)

        if res > 0.65:
            print(len(indicators))
            # print(i, j)
            data = data.rename({i: j}, axis=1)  # new method
            indicators.remove(j)

            if res < 0.68:
                print(res)
                print(i)
                print(j)
                print("------------------")

            break


print(data.columns)
data.to_csv("result_all_indicators_data2.csv", index=True , encoding="utf-8-sig")

# print(data.columns)