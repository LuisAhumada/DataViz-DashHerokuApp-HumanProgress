import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np


## 1- Load the Human Freedom Index data

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
# for key, value in dataframes.items():
#     value["hfi_score"] = ""
#     value["pf_score"] = ""
#     value["hfi_ef_score"] = ""
#     value["pf_rol"] = ""
#     value["pf_ss"] = ""
#     value["pf_movement"] = ""
#     value["pf_religion"] = ""
#     value["pf_association"] = ""
#     value["pf_expression"] = ""
#     value["pf_identity"] = ""
#     for f in range(len(value)):
#         for k in range(len(hfi)):
#             if key == hfi["countries"][k] and value["year"][f] == hfi["year"][k]:
#                 value.at[f, "hfi_score"] = hfi["hf_score"][k]
#                 value.at[f, "pf_score"] = hfi["ef_score"][k]
#                 value.at[f, "hfi_ef_score"] = hfi["pf_score"][k]
#                 value.at[f, "pf_rol"] = hfi["pf_rol"][k]
#                 value.at[f, "pf_ss"] = hfi["pf_ss"][k]
#                 value.at[f, "pf_movement"] = hfi["pf_movement"][k]
#                 value.at[f, "pf_religion"] = hfi["pf_religion"][k]
#                 value.at[f, "pf_association"] = hfi["pf_association"][k]
#                 value.at[f, "pf_expression"] = hfi["pf_expression"][k]
#                 value.at[f, "pf_identity"] = hfi["pf_identity"][k]
#
#
#
# # Read Economic Freedom of the World Index
# efw = pd.read_csv("/Users/luisabrigo/HumanProgress/efotw-2020.csv", header=0)
# print(efw.head())
# print(efw["Countries"])
#
# # Saves all EFW data in countries drataframes respectively
# for key, value in dataframes.items():
#     value["ef_score"] = ""
#     value["size_gov"] = ""
#     value["legal_property_rights"] = ""
#     value["sound_money"] = ""
#     value["trade"] = ""
#     value["regulation"] = ""
#     for f in range(len(value)):
#         for k in range(len(efw)):
#             if key == efw["Countries"][k] and value["year"][f] == efw["Year"][k]:
#                 value.at[f, "ef_score"] = efw["ef_score"][k]
#                 value.at[f, "size_gov"] = efw["size_gov"][k]
#                 value.at[f, "legal_property_rights"] = efw["legal_property_rights"][k]
#                 value.at[f, "sound_money"] = efw["sound_money"][k]
#                 value.at[f, "trade"] = efw["trade"][k]
#                 value.at[f, "regulation"] = efw["regulation"][k]
#
#
#
#

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
reg = linear_model.LinearRegression()

import os

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
result.to_csv("result_all_indicators.csv", index=True)



# 1. Indicator: GDP per Capita
# gdp = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/gdp-per-capita-2018-u.s.-dollars-1950–2019.csv", header=0)
#
# # print(gdp)
# # print(gdp["Country"][1])
# # print(gdp.iloc[4][2])
# # print(gdp.columns[45])
# # print(gdp["Albania"])
#
# gdp = pd.concat([gdp.iloc[:,0].astype(int), gdp.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
# # print(gdp)
#
# for key, value in dataframes.items():
#     if key in gdp.columns:
#         new = pd.merge(value, gdp[['year',key]],on='year', how='left')
#         value["gdp_per_capita"] = new[key]
#         # print(value)
#
#
#2. Poverty
# pov = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/poverty-headcount-ratio-at-1.90-a-day-percent-of-population-2011-international-dollars-ppp-1979–2016.csv", header=0)
#
# # print(pov.dtypes)
# # Converts float to object
# pov = pd.concat([pov.iloc[:,0].astype(int), pov.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
# # print(pov.columns)
# # print(pov.dtypes)
#
# for key, value in dataframes.items():
#     if key in pov.columns:
#         new = pd.merge(value, pov[['year',key]],on='year', how='left')
#         value["poverty"] = new[key]
#         # print(value)
#
#         value2 = value.dropna()
#         Y = value2["poverty"]
#         X = value2["year"].values.reshape(-1, 1)
#
#         reg = LinearRegression().fit(X, Y)
#         value["povertyPD"] = reg.predict(value.drop("poverty", axis=1))
#         value['povertyF'] = np.where(value['poverty'] > 0, value['poverty'], value['povertyPD'])
#         value = value.drop(['poverty', 'povertyPD'], axis=1, inplace=True)
# #
# for key, value in dataframes.items():
#     value.to_csv("countries_hfi/" + key + ".csv", index=False)



#
# #3. % Economic Growth
# #
# ec_grow = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/gdp-annual-growth-rate-percent-1961–2016.csv", header=0)
# # print(ec_grow)
#
# # print(ec_grow.iloc[:,0])
# # print(ec_grow.iloc[:,1:])
#
# # Converts float to object
# ec_grow = pd.concat([ec_grow.iloc[:,0].astype(int), ec_grow.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
# # print(ec_grow.columns)
# # print(ec_grow)
# # print(ec_grow.dtypes)
#
#
# for key, value in dataframes.items():
#     if key in ec_grow.columns:
#         new = pd.merge(value, ec_grow[['year',key]],on='year', how='left')
#         value["ec_growth"] = new[key]
#         # print(value)
# #
# # for key, value in dataframes.items():
# #     value.to_csv("countries_hfi/" + key + ".csv", index=False)
#
#
# #4. Life Expectancy
#
# life_exp = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/life-expectancy-at-birth-years-1960–2015.csv", header=0)
# life_exp = pd.concat([life_exp.iloc[:,0].astype(int), life_exp.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# # print(life_exp)

new_names = ["year", "Bahamas, The", "Brunei Darussalam", "Cabo Verde",  "Congo, Dem. Rep.", "Congo, Rep.", "Cote d'Ivoire", "Egypt, Arab Rep.", "Eswatini",
"Gambia, The", "Guinea-Bissau", "Hong Kong SAR, China", "Iran, Islamic Rep.", "Korea, Rep.", "Kyrgyz Republic", "Lao PDR", "Myanmar", "North Macedonia", "Russian Federation",
"Slovak Republic", "Syrian Arab Republic", "Timor-Leste", "Venezuela, RB", "Yemen, Rep."]

old_names = ["Country", "Bahamas", "Brunei", "Cape Verde",  "Congo (Kinshasa)", "Congo (Brazzaville)", "Ivory Coast", "Egypt", "Swaziland",
"Gambia", "Guinea Bissau", "Hong Kong", "Iran", "South Korea", "Kyrgyzstan", "Laos", "Burma (Myanmar)", "Macedonia (F.Y.R.O.M.)", "Russia",
"Slovakia", "Syria", "Timor Leste", "Venezuela", "Yemen"]

# # print(life_exp.columns)
# #
# for i in range(len(life_exp.columns)):
#     if life_exp.columns[i] in old_names:
#         a = life_exp.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         life_exp = life_exp.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
# # print(life_exp.columns)
#
# for key, value in dataframes.items():
#     if key in life_exp.columns:
#         new = pd.merge(value, life_exp[['year',key]],on='year', how='left')
#         value["life_exp"] = new[key]
#         # print(value)
#
#
#
# #5. Child Mortality
#
# child_mort = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/mortality-rate-children-under-5-per-1000-live-births-1960–2015.csv", header=0)
# child_mort = pd.concat([child_mort.iloc[:,0].astype(int), child_mort.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(child_mort.columns)):
#     if child_mort.columns[i] in old_names:
#         a = child_mort.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         child_mort = child_mort.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
# # print(child_mort.columns)
#
# for key, value in dataframes.items():
#     if key in child_mort.columns:
#         new = pd.merge(value, child_mort[['year',key]],on='year', how='left')
#         value["child_mort"] = new[key]
#
#
# #6. Vaccination
#
# vaccine = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/dtp3-diphtheria-tetanus-pertussis-vaccination-percent-of-children-aged-0-to-12-months-1980–2016.csv", header=0)
# vaccine = pd.concat([vaccine.iloc[:,0].astype(int), vaccine.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(vaccine.columns)):
#     if vaccine.columns[i] in old_names:
#         a = vaccine.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         vaccine = vaccine.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in vaccine.columns:
#         new = pd.merge(value, vaccine[['year',key]],on='year', how='left')
#         value["vaccine"] = new[key]
#
#
# #7. Homicide Rate
#
# homicide = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/homicide-rate-per-100000-1995–2011.csv", header=0)
# homicide = pd.concat([homicide.iloc[:,0].astype(int), homicide.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(homicide.columns)):
#     if homicide.columns[i] in old_names:
#         a = homicide.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         homicide = homicide.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in homicide.columns:
#         new = pd.merge(value, homicide[['year',key]],on='year', how='left')
#         value["homicide"] = new[key]
# #
#
#
#
# #9. Global Child Labor
#
# child_labor = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/economically-active-children-percent-of-children-aged-7-to-14-1994–2015.csv", header=0)
# child_labor = pd.concat([child_labor.iloc[:,0].astype(int), child_labor.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(child_labor.columns)):
#     if child_labor.columns[i] in old_names:
#         a = child_labor.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         child_labor = child_labor.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in child_labor.columns:
#         new = pd.merge(value, child_labor[['year',key]],on='year', how='left')
#         value["child_labor"] = new[key]
#
#
#
# #10. Human Development Index
#
# hdi = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/human-development-index-scale-0-1-1990–2015.csv", header=0)
# hdi = pd.concat([hdi.iloc[:,0].astype(int), hdi.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(hdi.columns)):
#     if hdi.columns[i] in old_names:
#         a = hdi.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         hdi = hdi.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in hdi.columns:
#         new = pd.merge(value, hdi[['year',key]],on='year', how='left')
#         value["hdi"] = new[key]
# #
#
#
#
# #11. Women's Freedom / Wage Gap
#
# wage_gap = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/gender-wage-gap-median-earnings-of-full-time-employees-oecd-percent-1970–2015.csv", header=0)
# wage_gap = pd.concat([wage_gap.iloc[:,0].astype(int), wage_gap.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(wage_gap.columns)):
#     if wage_gap.columns[i] in old_names:
#         a = wage_gap.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         wage_gap = wage_gap.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in wage_gap.columns:
#         new = pd.merge(value, wage_gap[['year',key]],on='year', how='left')
#         value["wage_gap"] = new[key]
# #
#
# #13 Life Satisfaction / Happiness
#
# life_satis = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/share-of-people-who-say-they-are-very-happy-or-quite-happy-percent-1981–2014.csv", header=0)
# life_satis = pd.concat([life_satis.iloc[:,0].astype(int), life_satis.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(life_satis.columns)):
#     if life_satis.columns[i] in old_names:
#         a = life_satis.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         life_satis = life_satis.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in life_satis.columns:
#         new = pd.merge(value, life_satis[['year',key]],on='year', how='left')
#         value["life_satis"] = new[key]
#
#
#
# #14. CO2 Emissions
#
# co2 = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/co2-emissions-per-capita-tonnes-1785–2016.csv", header=0)
# co2 = pd.concat([co2.iloc[:,0].astype(int), co2.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(co2.columns)):
#     if co2.columns[i] in old_names:
#         a = co2.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         co2 = co2.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in co2.columns:
#         new = pd.merge(value, co2[['year',key]],on='year', how='left')
#         value["co2"] = new[key]
# #
#
#
#
# #15. Average Year of Education
#
# edu = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/mean-years-of-primary-schooling-number-1870–2040.csv", header=0)
# edu = pd.concat([edu.iloc[:,0].astype(int), edu.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(edu.columns)):
#     if edu.columns[i] in old_names:
#         a = edu.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         edu = edu.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in edu.columns:
#         new = pd.merge(value, edu[['year',key]],on='year', how='left')
#         value["edu"] = new[key]
# #
#
#
#
# #18. Access to electricity (% population)
#
# elect = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/access-to-electricity-percent-of-population-1990–2014.csv", header=0)
# elect = pd.concat([elect.iloc[:,0].astype(int), elect.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(elect.columns)):
#     if elect.columns[i] in old_names:
#         a = elect.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         elect = elect.rename(columns={a: c})
#
#         # print(a)
#         # print(b)
#         # print(c)
#         # print("--------")
#
#
# for key, value in dataframes.items():
#     if key in elect.columns:
#         new = pd.merge(value, elect[['year',key]],on='year', how='left')
#         value["elect"] = new[key]

# from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
# reg = linear_model.LinearRegression()
#
#19. Access to Clean Water (%Population)

# clean_water = pd.read_csv("/Users/luisabrigo/HumanProgress/indicators/population-using-improved-drinking-water-sources-percent-1990–2015.csv", header=0)
# clean_water = pd.concat([clean_water.iloc[:,0].astype(int), clean_water.iloc[:,1:].astype(object)], join = 'outer', axis = 1)
#
# for i in range(len(clean_water.columns)):
#     if clean_water.columns[i] in old_names:
#         a = clean_water.columns[i]
#         b = old_names.index(a)
#         c = new_names[b]
#         clean_water = clean_water.rename(columns={a: c})
#
#
# for key, value in dataframes.items():
#     if key in clean_water.columns:
#         new = pd.merge(value, clean_water[['year',key]],on='year', how='left')
#         value["clean_water"] = new[key]
#
#         value2 = value.dropna()
#         Y = value2["clean_water"]
#         X = value2["year"].values.reshape(-1, 1)
#
#         reg = LinearRegression().fit(X, Y)
#         value["clean_waterPD"] = reg.predict(value.drop("clean_water", axis=1))
#         value['clean_water_F'] = np.where(value['clean_water'] > 0, value['clean_water'], value['clean_waterPD'])
#         value = value.drop(['clean_water', 'clean_waterPD'], axis=1, inplace=True)
#
# for key, value in dataframes.items():
#     value.to_csv("countries_hfi/" + key + ".csv", index=False)




#20. Sanitation
#21. Access to Internet


# names = []
# df_all = []
# for key, value in dataframes.items():
#     df_all.append(value)
#     names.append(key)

# result = pd.concat(df_all,keys=names)
# print(result)
# result.to_csv("result.csv", index=False)