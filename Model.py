import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.linalg as LA
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


## 1- Load the time series data called auto.csv

df = pd.read_csv("/Users/luisabrigo/HumanProgress/result.csv", header=0, parse_dates=[0])

df = df.dropna()


df_Y = df["life-expectancyNotFilledFilled"]

df_X = df.drop("life-expectancyNotFilledFilled", 1)
df_X = df_X.drop(labels=df.columns[:2], axis=1)

print(df_Y)


# print(df_X)

# print(df1.head())
# print(df1.dtypes)


#Train-test split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_X, df_Y, shuffle=True, test_size=0.2)

#Seaborn Heatmap
import seaborn as sns
corr = df_X.corr()
corr = df.corr()


plt.subplots(figsize=(15,15))
sns.heatmap(corr.T, annot=True, cmap="YlGnBu")
plt.tight_layout(pad=2, w_pad=0.5, h_pad=0.5)
plt.title("Correlation Matrix")
plt.show()

##Collinearity detection:
# a. Perform SVD analysis on the original feature space and write down your observation if colinearity exists. Justify your answer.
# b. Calculate the condition number and write down your observation if co-linearity exists

# print(df1.shape)

# #Converts dataframe to numpy matrix
# X = Xtrain.values
# print("Shape of X is:", X.shape)
# print(X)
#
# #Matrix multiplication
# H = np.matmul(X.T, X)
# print("Shape of H is:", H.shape)
#
# # Single Value Decomposition. If values are close to zero means the variables are correlated (multicolinearity)
# s, d, v = np.linalg.svd(H)
# print("Singular values:", d)
#
# #Condition Number
# print("Condition number for X is:", LA.cond(X))
# print("Colinearity is Severe")


# Using Python, construct matrix X and Y. Then use the x-train and y-train dataset and estimate the
# regression model unknown coefficients using the Normal equation (LSE method, above equation)



# Converts train set to numpy matrix
# Xt = Xtrain.values
# Yt = Ytrain.values
#
# print("Shape of Xtrain is:", Xt.shape)
# print("----------")
#
# print("Shape of Ytrain is:", Yt.shape)

# Ordinary least square regression using Numpy Linalg
# inv = np.linalg.inv(H)
# beta = np.matmul(np.matmul(inv, Xt.T), Yt)
# print("Betas:", beta)



# model = sm.OLS(Ytrain.astype(float), Xtrain.astype(float)).fit()
# print(model.summary())

# Adds a column called "const"
Xtrain = sm.add_constant(Xtrain)
Xtest = sm.add_constant(Xtest)





#Using python, statsmodels package and OLS function, find the unknown coefficients. C

model = sm.OLS(Ytrain.astype(float), Xtrain.astype(float)).fit()
print(model.summary())

#Feature selection: Using a backward stepwise regression, reduce the feature space
# dimension. You need to use the AIC, BIC and Adjusted R2 as a predictive accuracy for your analysis.

# remove economically-acNotFilledFilled
x_opt = Xtrain.drop("economically-acNotFilledFilled", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove human-developmeNotFilledFilled
x_opt = x_opt.drop("human-developmeNotFilledFilled", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())


# # remove share-of-peopleNotFilledFille
x_opt = x_opt.drop("share-of-peopleNotFilledFilled", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove co2-emissions-pNotFilledFilled
x_opt = x_opt.drop("co2-emissions-pNotFilledFilled", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())

# # remove gdp-annual-growNotFilledFilled
x_opt = x_opt.drop("gdp-annual-growNotFilledFilled", 1)
ols = sm.OLS(Ytrain, x_opt).fit()
print(ols.summary())


# Edit the test set
Xtest_new = Xtest[["const", "dtp3-diphtheriaNotFilledFilled",
    "population-usinNotFilledFilled", "homicide-rate-pNotFilledFilled",
    "gdp-per-capita-NotFilledFilled", "mean-years-of-pNotFilledFilled",
    "mortality-rate-NotFilledFilled", "access-to-electNotFilledFilled", "poverty-headcouNotFilledFilled"]]


Ypred = ols.predict(x_opt)

Yforecast = ols.predict(Xtest_new)
print(Yforecast)


# plt.plot(Xtrain.index, Ytrain, label="y")
# plt.plot(Xtest.index, Ytest, label="Y_test")
# plt.plot(Xtest.index, Yforecast, label="Y_forecast")
# plt.legend(loc='best')
# plt.xlabel("Index")
# plt.ylabel("Price")
# plt.title("Price of cars")
# plt.show()
#"
def autocorr(x, lag):
    '''It returns the autocorrelation function for a given series.'''
    x = np.array(x)

    ## Proceed to calculate the numerator only for the oservationes given by the length of lag
    k = range(0,lag+1)

    ## Calculates the mean of the obs in the variable
    mean = np.mean(x)

    autocorr = []
    #Calculates a numerator and denominator based on the autocorrelation function formula
    for i in k:
        numer = 0
        for j in range(i, len(x)):
            numer += (x[j] - mean) * (x[j-i] - mean)
        denom = np.sum((x - mean) ** 2)
        autocorr.append(numer / denom)
    return autocorr

def ACFPlot(x, lags, title):
    # Limits the plot to the amount of lags given
    x = x[0:lags+1]
    # Makes the values go forward and backward (positive and negative)
    rx = x[::-1]
    rxx = np.concatenate((rx, x[1:]))
    lags = [i for i in range(-lags, lags+1)]
    lags = np.unique(np.sort(lags))
    plt.figure()
    plt.stem(lags, rxx)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.title(title)
    plt.axhspan(-(x[4]), x[4], alpha=.1, color='black')
    plt.show()

def variance(x, k):
    var = np.sqrt((1/(len(x)-k-1))*(np.sum(np.square(x))))
    return var

#Calculates prediciton and forecast error
p_error = Ytrain - Ypred
print("This is the prediction error:", p_error)
print("This is the prediction error variance:", variance(p_error, 4))

f_error = Ytest - Yforecast
print("This is the forecast error:", f_error)
print("This is the forecast error variance:", variance(f_error, 4))

# ACF Plots
ACF_p_error = autocorr(p_error, 20)
ACFPlot(ACF_p_error, 20, "ACF Prediction Error")
ACF_f_error = autocorr(f_error, 20)
ACFPlot(ACF_f_error, 20, "ACF Forecast Error")


def MSE(x):
    return np.sum(np.power(x, 2)) / len(x)


MSE = MSE(f_error)
print("MSE Prediction Error: ", round(MSE, 3))


# Calculate R2
def R2(yhat, y_test):
    y_mean = np.mean(y_test)
    r2 = np.sum((yhat - y_mean) ** 2) / np.sum((y_test - y_mean) ** 2)
    return r2


R2 = R2(Yforecast, Ytest)
print("The R^2 coefficient is:", round(R2, 3))


#

# import statsmodels.formula.api as smf
#
# def forward_selected(data, response):
#     """Linear model designed by forward selection.
#
#     Parameters:
#     -----------
#     data : pandas DataFrame with all possible predictors and response
#
#     response: string, name of response column in data
#
#     Returns:
#     --------
#     model: an "optimal" fitted statsmodels linear model
#            with an intercept
#            selected by forward selection
#            evaluated by adjusted R-squared
#     """
#     remaining = set(data.columns)
#     remaining.remove(response)
#     selected = []
#     current_score, best_new_score = 0.0, 0.0
#     while remaining and current_score == best_new_score:
#         scores_with_candidates = []
#         for candidate in remaining:
#             formula = "{} ~ {} + 1".format(response,
#                                            ' + '.join(selected + [candidate]))
#             score = smf.ols(formula, data).fit().rsquared_adj
#             scores_with_candidates.append((score, candidate))
#         scores_with_candidates.sort()
#         best_new_score, best_candidate = scores_with_candidates.pop()
#         if current_score < best_new_score:
#             remaining.remove(best_candidate)
#             selected.append(best_candidate)
#             current_score = best_new_score
#     formula = "{} ~ {} + 1".format(response,
#                                    ' + '.join(selected))
#     model = smf.ols(formula, data).fit()
#     return model
#
# model = forward_selected(df1, "price")
# print(model.model.formula)
# print(model.rsquared_adj)

