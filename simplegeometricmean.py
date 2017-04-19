"""
Private Score: 0.15996, Public Score: 0.15390
"""

import datetime as dt

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import scipy.stats

pd.options.mode.chained_assignment = None


################################################################
# Import CSV Data into Pandas DataFrames                       #
################################################################
training_df = pd.read_csv("data/train.csv")
# store_df = pd.read_csv("data/store.csv")
test_df = pd.read_csv("data/test.csv")

# print(training_df.head())
# print(store_df.head())
# print(test_df.head())


################################################################
# Process Data (Universal)                                     #
################################################################

def is_nan(val):
    return val != val

############################################
# training_df & test_df                    #
############################################

# Fill NaN values in test_df with Open = 1 if DayOfWeek != 7
test_df["Open"][is_nan(test_df["Open"])] = (test_df["DayOfWeek"] != 7).astype(int)

# Create "Year" & "Month" columns
# training_df["Year"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year)
# training_df["Month"] = training_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month)

# test_df["Year"] = test_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").year)
# test_df["Month"] = test_df["Date"].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").month)

# Create "YearMonth" column
# training_df["YearMonth"] = training_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + str(dt.datetime.strptime(x, "%Y-%m-%d").month))
# test_df["YearMonth"] = test_df["Date"].apply(lambda x: str(dt.datetime.strptime(x, "%Y-%m-%d").year) + "-" + str(dt.datetime.strptime(x, "%Y-%m-%d").month))

# "StateHoliday" has values "0" & 0
# training_df["StateHoliday"].loc[training_df["StateHoliday"] == 0] = "0"
# test_df["StateHoliday"].loc[test_df["StateHoliday"] == 0] = "0"

# Create "StateHolidayBinary" column
# training_df["StateHolidayBinary"] = training_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
# test_df["StateHolidayBinary"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

# One-hot encoding of "DayOfWeek" & "StateHoliday" columns
# training_df = pd.get_dummies(training_df, columns=["DayOfWeek", "StateHoliday"])
# test_df = pd.get_dummies(test_df, columns=["DayOfWeek", "StateHoliday"])

############################################
# store_df                                 #
############################################

# Fill NaN values in store_df for "CompetitionDistance" = 0 (since no record exists where "CD" = NaN & "COS[Y/M]" = !NaN)
# store_df["CompetitionDistance"][is_nan(store_df["CompetitionDistance"])] = 0

# Fill NaN values in store_df for "CompetitionSince[X]" with 1900-01
# store_df["CompetitionOpenSinceYear"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceYear"]))] = 1900
# store_df["CompetitionOpenSinceMonth"][(store_df["CompetitionDistance"] != 0) & (is_nan(store_df["CompetitionOpenSinceMonth"]))] = 1

# One-hot encoding of "StoreType" & "Assortment" columns
# store_df = pd.get_dummies(store_df, columns=["StoreType", "Assortment"])


################################################################
# Process Data (Custom)                                        #
################################################################

# Any custom data processing goes here.


################################################################
# Training the Model & Predicting Sales                        #
################################################################

"""
This model simply calculates the geometric mean value for every ["Store", "DayOfWeek", "Promo"] combination and assigns that value as the prediction for every ["Store", "DayOfWeek", "Promo"] combination in the test data.

Features: Store, DayOfWeek, Promo

Assumptions:
- The only factors that significantly affect the sales in a particular store are "DayOfWeek" & "Promo".
"""

columns = ["Store", "DayOfWeek", "Promo"]

geomeans = pd.concat([training_df[(training_df.Sales != 0) & (training_df.DayOfWeek != 7)].groupby(columns).agg({"Sales": lambda x: np.exp(scipy.stats.mstats.gmean(np.log(x + 1))) - 1}), training_df[training_df.DayOfWeek == 7].groupby(columns).agg({"Sales": lambda x: np.exp(scipy.stats.mstats.gmean(np.log(x + 1))) - 1})])
geomeans = geomeans.reset_index()

test_df_modified = pd.merge(test_df, geomeans, on=columns, how="left")
test_df_modified.loc[test_df_modified.Open == 0, "Sales"] = 0

test_df_modified[["Id", "Sales"]].to_csv("predictions/simplegeometricmean.csv", index=False)
