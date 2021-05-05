import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np


## 1- Load the Human Freedom Index data to extract country names

hfi = pd.read_csv("/Users/luisabrigo/HumanProgress/hfi.csv", header=0)

# print(df.head())
# print(df["countries"])

# Creates a list with the name of the countries
countries = []
dataframes = {}
for i in range(len(hfi)):
    if hfi["countries"][i] not in countries:
        countries.append(hfi["countries"][i])


#Initialize a dictionary with only keys from a list
for i in countries:
    #All countriess dataframes are saved in dict dataframes
    dataframes[i] = i


# #Creates a dataframe for each country with a column "year" from 1960 to 2020.
for i in dataframes:
    df = pd.DataFrame(pd.Series(np.arange(1960,2021,1)),
                   columns=['year'])

    #All countriess dataframes are saved in dict dataframes
    dataframes[i] = df



print(countries)

nums = []
for i in range(162):
    i = str(i)
    i = i.zfill(3)
    nums.append(i)

controles = dict(zip(nums, countries))

import json
# A dictionary of student names and their score
# Print contents of dict in json like format
print(json.dumps(controles, indent=4))

# # Saves all HFI data in countries drataframes respectively
for key, value in dataframes.items():
    value["hfi_score"] = ""
    value["pf_score"] = ""
    value["hfi_ef_score"] = ""
    value["pf_rol"] = ""
    value["pf_ss"] = ""
    value["pf_movement"] = ""
    value["pf_religion"] = ""
    value["pf_association"] = ""
    value["pf_expression"] = ""
    value["pf_identity"] = ""
    for f in range(len(value)):
        for k in range(len(hfi)):
            if key == hfi["countries"][k] and value["year"][f] == hfi["year"][k]:
                value.at[f, "hfi_score"] = hfi["hf_score"][k]
                value.at[f, "pf_score"] = hfi["ef_score"][k]
                value.at[f, "hfi_ef_score"] = hfi["pf_score"][k]
                value.at[f, "pf_rol"] = hfi["pf_rol"][k]
                value.at[f, "pf_ss"] = hfi["pf_ss"][k]
                value.at[f, "pf_movement"] = hfi["pf_movement"][k]
                value.at[f, "pf_religion"] = hfi["pf_religion"][k]
                value.at[f, "pf_association"] = hfi["pf_association"][k]
                value.at[f, "pf_expression"] = hfi["pf_expression"][k]
                value.at[f, "pf_identity"] = hfi["pf_identity"][k]



# Read Economic Freedom of the World Index
efw = pd.read_csv("/Users/luisabrigo/HumanProgress/efotw-2020.csv", header=0)
print(efw.head())
print(efw["Countries"])

# Saves all EFW data in countries drataframes respectively
for key, value in dataframes.items():
    value["ef_score"] = ""
    value["size_gov"] = ""
    value["legal_property_rights"] = ""
    value["sound_money"] = ""
    value["trade"] = ""
    value["regulation"] = ""
    for f in range(len(value)):
        for k in range(len(efw)):
            if key == efw["Countries"][k] and value["year"][f] == efw["Year"][k]:
                value.at[f, "ef_score"] = efw["ef_score"][k]
                value.at[f, "size_gov"] = efw["size_gov"][k]
                value.at[f, "legal_property_rights"] = efw["legal_property_rights"][k]
                value.at[f, "sound_money"] = efw["sound_money"][k]
                value.at[f, "trade"] = efw["trade"][k]
                value.at[f, "regulation"] = efw["regulation"][k]





from sklearn import linear_model
from sklearn.linear_model import LinearRegression
reg = linear_model.LinearRegression()

import os

#Rename countries
new_names = ["year", "Bahamas, The", "Brunei Darussalam", "Cabo Verde",  "Congo, Dem. Rep.", "Congo, Rep.", "Cote d'Ivoire", "Egypt, Arab Rep.", "Eswatini",
"Gambia, The", "Guinea-Bissau", "Hong Kong SAR, China", "Iran, Islamic Rep.", "Korea, Rep.", "Kyrgyz Republic", "Lao PDR", "Myanmar", "North Macedonia", "Russian Federation",
"Slovak Republic", "Syrian Arab Republic", "Timor-Leste", "Venezuela, RB", "Yemen, Rep."]

old_names = ["Country", "Bahamas", "Brunei", "Cape Verde",  "Congo (Kinshasa)", "Congo (Brazzaville)", "Ivory Coast", "Egypt", "Swaziland",
"Gambia", "Guinea Bissau", "Hong Kong", "Iran", "South Korea", "Kyrgyzstan", "Laos", "Burma (Myanmar)", "Macedonia (F.Y.R.O.M.)", "Russia",
"Slovakia", "Syria", "Timor Leste", "Venezuela", "Yemen"]


directory = "/Users/luisabrigo/HumanProgress/all_indicators"



for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        print(filename)
        short_name = filename[:-4] + "NF-"
        print(short_name)
        try:
            df = pd.read_csv(directory + "/" + filename, header=0)
            if len(df) > 10:
                df = df.transpose()
                df.reset_index(level=0, inplace=True)
                df = df.rename(columns=df.iloc[0]).drop(df.index[0])
                df.reset_index(inplace=True)
                df = df.drop(['index'], axis=1)
                df = df.rename(columns={"Country": "year"})
                df = df.rename(columns={"Data Item": "year"})
                df = df.replace({'..': None})
                df = df.replace({'--': None})
                df = pd.concat([df.iloc[:,0].astype(int), df.iloc[:,1:].astype(object)], join = 'outer', axis = 1)

                for i in range(len(df.columns)):
                    if df.columns[i] in old_names:
                        a = df.columns[i]
                        b = old_names.index(a)
                        c = new_names[b]
                        df = df.rename(columns={a: c})



                for key, value in dataframes.items():
                    if key in df.columns:
                        #If the indicator contains more than 6 values (rows)
                        if df[key].count() > 6:

                            new = pd.merge(value, df[['year',key]],on='year', how='left')
                            value[short_name] = new[key]


                            value0 = value[["year", short_name]]
                            value2 = value0.dropna()
                            Y = value2[short_name]
                            X = value2["year"].values.reshape(-1, 1)

                            step2 = short_name + "PD"
                            step3 = short_name + "F-"

                            try:
                                reg = LinearRegression().fit(X, Y)
                                value[step2] = reg.predict(value["year"].values.reshape(-1, 1))
                                value[step3] = np.where(value[short_name] > 0, value[short_name], value[step2])
                                value = value.drop([step2], axis=1, inplace=True)

                            except Exception as e:
                                print(e, key, value, filename)


                        else:
                            print("There are not enough values for " + key + " on " + filename)
            else:
                print("Not enough rows in DF")
        except Exception as e:
            print(e)

for key, value in dataframes.items():
    value.to_csv("countries_hfi/" + key + ".csv", index=False)

    #Save only filled variables
    value = value.set_index('year')
    valueFilled = value.loc[:, ~value.columns.str.endswith('NF-')]
    valueFilled.to_csv("all_countries_indicators/" + key + ".csv", index=True)



### Merging all countries
names = []
df_all = []
for key, value in dataframes.items():
    value = value.set_index('year')
    valueFilled = value.loc[:, ~value.columns.str.endswith('NF-')]

    df_all.append(valueFilled)
    names.append(key)

result = pd.concat(df_all,keys=names)
print(result)
result.to_csv("df_all.csv", index=True)



